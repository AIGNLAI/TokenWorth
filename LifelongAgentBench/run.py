import argparse
import json
import os
import yaml
import copy
import signal
from pathlib import Path

from src.utils.checkpoint import CheckpointManager, atomic_json, run_lock
from src.utils.injection_budget import enforce_injection_budget
from src.utils.runtime import (build_runtime_manifest, ensure_runtime_manifest, invalid_session_reason, sanitize, validate_prefix, validate_sample_order, validate_no_replay, UncommittedInferenceError)
from typing import Any, Mapping, Sequence, Optional
import coredumpy  # type: ignore[import-untyped]

from src.utils import ConfigLoader, SingletonLogger
from src.typings import (
    AssignmentConfig,
    SampleStatus,
    LoggerConfig,
    ContinualAgentBenchException,
    Session,
    SampleIndex,
    PathConfig,
    GeneralInstanceFactory,
    SessionMetricCalculationPartial,
)
from src.tasks import Task, DatasetItem
from src.agents import Agent
from src.language_models import LanguageModel
from src.callbacks import (
    CallbackHandler,
    CallbackConstructor,
    Callback,
    CallbackRestorer,
    CallbackArguments,
)


def _resolve_language_model_override_name(
    raw_config: Mapping[str, Any], requested_name: Optional[str]
) -> str:
    language_model_name_list = [
        language_model_info_dict["name"]
        for language_model_info_dict in raw_config["assignment_config"][
            "language_model_list"
        ]
    ]
    if requested_name is not None:
        if requested_name not in language_model_name_list:
            raise ValueError(
                f"Language model {requested_name!r} is not used by this assignment. "
                f"Available language models: {language_model_name_list}."
            )
        return requested_name
    if len(language_model_name_list) == 1:
        return language_model_name_list[0]
    agent_language_model_name = (
        raw_config["assignment_config"]
        .get("agent", {})
        .get("custom_parameters", {})
        .get("language_model")
    )
    if agent_language_model_name in language_model_name_list:
        return agent_language_model_name
    raise ValueError(
        "Multiple language models are used by this assignment. "
        "Please specify one with --llm-name."
    )


def apply_language_model_overrides(
    raw_config: dict[str, Any],
    language_model_name: Optional[str],
    model_name: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    maximum_prompt_token_count: Optional[int],
    timeout: Optional[float],
    max_retries: Optional[int],
    trust_env: Optional[bool],
) -> dict[str, Any]:
    override_dict = {
        "model_name": model_name,
        "base_url": base_url,
        "api_key": api_key,
        "maximum_prompt_token_count": maximum_prompt_token_count,
        "timeout": timeout,
        "max_retries": max_retries,
        "trust_env": trust_env,
    }
    override_dict = {
        key: value for key, value in override_dict.items() if value is not None
    }
    if not override_dict:
        return raw_config
    raw_config = copy.deepcopy(raw_config)
    resolved_language_model_name = _resolve_language_model_override_name(
        raw_config, language_model_name
    )
    language_model_parameters = raw_config["language_model_dict"][
        resolved_language_model_name
    ].setdefault("parameters", {})
    language_model_parameters.update(override_dict)
    return raw_config


class ConfigUtility:
    def __init__(
        self,
        assignment_config: AssignmentConfig,
        path_config: PathConfig,
    ):
        self.assignment_config = assignment_config
        self.path_config = path_config

    def construct(self) -> tuple[Task[DatasetItem], Agent, dict[str, Callback]]:
        task: Task[DatasetItem] = self.assignment_config.task.create()
        # region Construct language_model_dict
        # Here, We actually instantiate the language models.
        # After the code exit construct(), We will never get a chance to get the language model instance.
        # But I think this is good, since this improve the modularity and maintainability of the code.
        try:
            language_model_dict: Mapping[str, LanguageModel] = {
                key: value.create()
                for key, value in self.assignment_config.language_model_dict.items()
            }
        except BaseException as error:
            try:
                task.release()
            except BaseException as cleanup_error:
                error.add_note(f"Task cleanup also failed: {cleanup_error}")
            raise
        self.language_model_dict = language_model_dict
        try:
            agent_instance_factory: GeneralInstanceFactory = self.assignment_config.agent
            if (
                language_model_name := agent_instance_factory.parameters.get(
                    "language_model"
                )
            ) is not None:
                agent_instance_factory.parameters["language_model"] = language_model_dict[
                    language_model_name
                ]
            # endregion
            agent: Agent = agent_instance_factory.create()
            callback_dict = CallbackConstructor.construct(
                self.assignment_config, task, agent, language_model_dict
            )
            return task, agent, callback_dict
        except BaseException as error:
            try:
                task.release()
            except BaseException as cleanup_error:
                error.add_note(f"Task cleanup also failed: {cleanup_error}")
            raise

    def validate(self, task: Task[DatasetItem], agent: Agent) -> None:
        validate_sample_order(self.assignment_config.sample_order, task.get_sample_index_list())

    def postprocess(self, task: Task[DatasetItem], agent: Agent) -> None:
        if self.assignment_config.sample_order == "default":
            self.assignment_config.sample_order = task.get_sample_index_list()

    def remove_redundant_args(self, raw_config: dict[str, Any]) -> dict[str, Any]:
        redundant_key_buffer: set[tuple[str, str]] = set()
        for key in raw_config["task_dict"]:
            if key != raw_config["assignment_config"]["task"]:
                redundant_key_buffer.add(("task_dict", key))
        for key in raw_config["agent_dict"]:
            if key != raw_config["assignment_config"]["agent"]["name"]:
                redundant_key_buffer.add(("agent_dict", key))
        assignment_language_model_name_list: Sequence[str] = [
            language_model_info_dict["name"]
            for language_model_info_dict in raw_config["assignment_config"][
                "language_model_list"
            ]
        ]
        for key in raw_config["language_model_dict"]:
            if key not in assignment_language_model_name_list:
                redundant_key_buffer.add(("language_model_dict", key))
        assignment_callback_name_list: Sequence[str] = [
            callback_info_dict["name"]
            for callback_info_dict in raw_config["assignment_config"][
                "callback_dict"
            ].values()
        ]
        for key in raw_config["callback_dict"]:
            if key not in assignment_callback_name_list:
                redundant_key_buffer.add(("callback_dict", key))
        for info_tuple in redundant_key_buffer:
            del raw_config[info_tuple[0]][info_tuple[1]]
        return raw_config

    @staticmethod
    def _get_custom_instance_info_dict(
        default_instance_info_dict: Mapping[str, Any],
        custom_instance_info_dict: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        module: str = default_instance_info_dict["module"]
        # `... or {}` is used to ensure that both default_parameters and custom_parameters are dict.
        default_parameters: dict[str, Any] = copy.deepcopy(
            default_instance_info_dict.get("parameters") or {}
        )
        custom_parameters = custom_instance_info_dict.get("custom_parameters") or {}
        for parameter_name, custom_parameter_value in custom_parameters.items():
            assert parameter_name in default_parameters  # Do not remove this assertion.
            # Overwrite the default parameter value with the custom parameter value.
            default_parameters[parameter_name] = custom_parameter_value
        return {
            "module": module,
            "parameters": default_parameters,
        }

    @staticmethod
    def read_raw_config(
        raw_config: Mapping[str, Any],
    ) -> tuple[AssignmentConfig, LoggerConfig, PathConfig]:
        raw_config = copy.deepcopy(raw_config)  # Avoid modifying the original config.
        # region Convert raw_config into assignment_config
        # region Construct assignment_language_model_dict
        assignment_language_model_list: Sequence[Mapping[str, Any]] = raw_config[
            "assignment_config"
        ]["language_model_list"]
        assignment_language_model_dict: dict[str, Any] = {}
        for language_model_info_dict in assignment_language_model_list:
            language_model_name = language_model_info_dict["name"]
            default_language_model_info_dict = raw_config["language_model_dict"][
                language_model_name
            ]
            assert language_model_name not in assignment_language_model_dict
            assignment_language_model_dict[language_model_name] = (
                ConfigUtility._get_custom_instance_info_dict(
                    default_language_model_info_dict, language_model_info_dict
                )
            )
        # endregion
        # region Construct assignment_agent
        custom_agent_info_dict = raw_config["assignment_config"]["agent"]
        default_agent_info_dict = raw_config["agent_dict"][
            custom_agent_info_dict["name"]
        ]
        assignment_agent_info_dict = ConfigUtility._get_custom_instance_info_dict(
            default_agent_info_dict, custom_agent_info_dict
        )
        assignment_agent = GeneralInstanceFactory.model_validate(
            assignment_agent_info_dict
        )
        if (
            language_model_name := assignment_agent.parameters.get("language_model")
        ) is not None:
            # Do not replace the language_model in the parameters with the GeneralInstanceFactory instance.
            assert language_model_name in assignment_language_model_dict
        # endregion
        # region Construct assignment_callback_dict
        assignment_callback_dict: dict[str, Any] = raw_config["assignment_config"][
            "callback_dict"
        ]
        for callback_key, callback_info_dict in assignment_callback_dict.items():
            default_callback_info_dict = raw_config["callback_dict"][
                callback_info_dict["name"]
            ]
            assignment_callback_dict[callback_key] = (
                ConfigUtility._get_custom_instance_info_dict(
                    default_callback_info_dict, callback_info_dict
                )
            )
        # endregion
        assignment_config = AssignmentConfig(
            task=raw_config["task_dict"][raw_config["assignment_config"]["task"]],
            agent=assignment_agent,
            language_model_dict=assignment_language_model_dict,
            output_dir=raw_config["assignment_config"]["output_dir"],
            sample_order=raw_config["assignment_config"]["sample_order"],
            callback_dict=assignment_callback_dict,
            generation_budget=raw_config["assignment_config"].get("generation_budget"),
        )
        # endregion
        # region Convert raw_config into logger_config
        if raw_config["logger_config"]["log_file_path"] == "default":
            log_file_path = os.path.join(
                assignment_config.output_dir, "singleton_logger.log"
            )
        else:
            log_file_path = raw_config["logger_config"]["log_file_path"]
        logger_config = LoggerConfig(
            level=raw_config["logger_config"]["level"],
            log_file_path=log_file_path,
            logger_name=raw_config["logger_config"]["logger_name"],
        )
        # endregion
        # region Construct path_config from assignment_config
        path_config = PathConfig(
            exception_record_file_path=os.path.join(
                assignment_config.output_dir, "exception.txt"
            ),
            config_output_path=os.path.join(
                assignment_config.output_dir, "config.yaml"
            ),
            session_list_output_path=os.path.join(
                assignment_config.output_dir, "runs.json"
            ),
            metric_output_path=os.path.join(
                assignment_config.output_dir, "metric.json"
            ),
            coredumpy_output_dir=os.path.join(
                assignment_config.output_dir, "coredumpy"
            ),
        )
        # endregion
        return assignment_config, logger_config, path_config

    @staticmethod
    def is_raw_config_equal(
        raw_config_1: dict[str, Any], raw_config_2: dict[str, Any]
    ) -> bool:
        raw_config_1 = copy.deepcopy(raw_config_1)
        raw_config_2 = copy.deepcopy(raw_config_2)
        output_dir_1 = raw_config_1["assignment_config"].pop("output_dir")
        output_dir_2 = raw_config_2["assignment_config"].pop("output_dir")
        return raw_config_1 == raw_config_2 and AssignmentConfig.is_output_dir_equal(
            output_dir_1, output_dir_2
        )


def main() -> None:
    # region Prepare variables
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override assignment_config.output_dir.",
    )
    parser.add_argument(
        "--llm-name",
        type=str,
        default=None,
        help=(
            "Name of the assignment language model to override. "
            "If omitted, the only language model in the assignment is used."
        ),
    )
    parser.add_argument(
        "--llm-model-name",
        type=str,
        default=None,
        help="Override OpenAI-compatible model name.",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help=(
            "Override OpenAI-compatible API base URL, "
            "for example http://localhost:8001/v1."
        ),
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help=(
            "Override API key. Use env:OPENAI_API_KEY to read it from the environment "
            "without writing the secret to the saved config."
        ),
    )
    parser.add_argument(
        "--llm-maximum-prompt-token-count",
        type=int,
        default=None,
        help="Override the maximum prompt token count guard for the language model.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=None,
        help="Override OpenAI client timeout in seconds.",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=None,
        help="Override OpenAI client automatic retry count.",
    )
    parser.add_argument(
        "--llm-ignore-env-proxy",
        action="store_true",
        help="Ignore proxy settings from http_proxy/https_proxy environment variables.",
    )
    parser.add_argument(
        "--kg-sparql-url",
        type=str,
        default=None,
        help="Override the SPARQL endpoint used by the knowledge-graph task.",
    )
    parser.add_argument("--db-image", type=str, default=None, help="Override the DB container image; its resolved ID is locked in the runtime manifest.")
    parser.add_argument("--os-image", type=str, default=None, help="Override the OS container image; its resolved ID is locked in the runtime manifest.")
    args = parser.parse_args()
    raw_config = ConfigLoader().load_from(args.config_path)
    if args.output_dir is not None:
        raw_config["assignment_config"]["output_dir"] = args.output_dir
    if args.kg_sparql_url is not None:
        raw_config["task_dict"]["knowledge_graph"]["parameters"][
            "sparql_url"
        ] = args.kg_sparql_url
    for task_name, image in (("db_bench", args.db_image), ("os_interaction", args.os_image)):
        if image is not None:
            raw_config["task_dict"][task_name]["parameters"]["container_image"] = image
    raw_config = apply_language_model_overrides(
        raw_config=raw_config,
        language_model_name=args.llm_name,
        model_name=args.llm_model_name,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        maximum_prompt_token_count=args.llm_maximum_prompt_token_count,
        timeout=args.llm_timeout,
        max_retries=args.llm_max_retries,
        trust_env=False if args.llm_ignore_env_proxy else None,
    )
    assignment_config, logger_config, path_config = ConfigUtility.read_raw_config(
        raw_config
    )
    # Prepare the variable that will be used in the following procedures.
    config_utility = ConfigUtility(assignment_config, path_config)
    # endregion
    def terminate(signum, frame):
        raise KeyboardInterrupt("Experiment interrupted by SIGTERM.")
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        with run_lock(assignment_config.output_dir):
            execute_assignment(raw_config, config_utility, logger_config, path_config)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def execute_assignment(raw_config, config_utility, logger_config, path_config):
    assignment_config = config_utility.assignment_config
    output = Path(assignment_config.output_dir)
    checkpoint = CheckpointManager(str(output))
    task = None
    session_list = []
    had_runtime_manifest = (output / "runtime_manifest.json").exists()
    identity_verified = False
    execution_error = None
    expected_count = len(assignment_config.sample_order) if isinstance(assignment_config.sample_order, list) else None
    def status(state, reasons=None):
        atomic_json(output / "run_status.json", {
            "schema_version": 1, "status": state, "valid": state == "completed",
            "expected_sample_count": expected_count, "completed_sample_count": len(session_list),
            "invalid_reasons": reasons or [], "checkpoint_generation": checkpoint.generation,
        })
    try:
        cleaned_config = sanitize(config_utility.remove_redundant_args(raw_config))
        config_path = Path(path_config.config_output_path)
        if config_path.exists():
            with config_path.open() as stream:
                previous = yaml.safe_load(stream)
            if not ConfigUtility.is_raw_config_equal(previous, cleaned_config):
                raise ValueError("Saved experiment config differs; resume refused.")
        else:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            # Sanitized config is for inspection; effective identity is locked below.
            temporary = config_path.with_suffix(".yaml.tmp")
            temporary.write_text(yaml.safe_dump(cleaned_config))
            os.replace(temporary, config_path)
        logger = SingletonLogger.get_instance(logger_config)
        coredumpy.patch_except(directory=path_config.coredumpy_output_dir)
        task, agent, callback_dict = config_utility.construct()
        config_utility.postprocess(task, agent)
        config_utility.validate(task, agent)
        expected_count = len(assignment_config.sample_order)
        if assignment_config.generation_budget is None:
            assignment_config.generation_budget = int(getattr(agent, "_inference_config_dict", {}).get("max_tokens", 128))
        if len(config_utility.language_model_dict) != 1:
            raise ValueError("The paper protocol requires one shared frozen backbone and generation budget.")
        manifest = ensure_runtime_manifest(str(output), build_runtime_manifest(
            raw_config, assignment_config, config_utility.language_model_dict,
            Path(__file__).resolve().parent, task_instance=task,
        ))
        identity_verified = True
        for alias, model in config_utility.language_model_dict.items():
            configure = getattr(model, "configure_audit", None)
            if configure:
                configure(output_dir=str(output), model_alias=alias, run_id=manifest["run_id"])
        ContinualAgentBenchException.set_record_file(path_config.exception_record_file_path)
        restored = checkpoint.restore()
        callback_handler = CallbackHandler(callback_dict)
        if restored is not None:
            session_list = [Session.model_validate(item) for item in restored]
            validate_prefix(session_list, assignment_config.sample_order, task.task_name)
            validate_no_replay(str(output), session_list)
            CallbackRestorer.restore(callback_dict)
        else:
            validate_no_replay(str(output), [])
            for callback in callback_dict.values():
                Path(callback.get_state_dir()).mkdir(parents=True, exist_ok=True)
            initial = Session(task_name=task.task_name, sample_index=assignment_config.sample_order[0])
            callback_handler.on_state_save(CallbackArguments(initial, task, agent, []))
            checkpoint.commit([])
        status("running")
        logger.info(f"Experiment start. Total sample count: {expected_count}. Unfinished sample count: {expected_count - len(session_list)}.")
        for sample_index in assignment_config.sample_order[len(session_list):]:
            for model in config_utility.language_model_dict.values():
                begin = getattr(model, "begin_sample", None)
                if begin:
                    begin(str(sample_index), assignment_config.generation_budget)
            session = Session(task_name=task.task_name, sample_index=sample_index)
            callback_args = CallbackArguments(session, task, agent, session_list)
            callback_handler.on_session_create(callback_args)
            if callback_args.session_controller.should_task_reset:
                task.reset(session)
                reason = invalid_session_reason(session, evaluated=False)
                if reason:
                    raise RuntimeError(reason)
                injection_chars = raw_config['assignment_config'].get('resource_budget', {}).get('injection_chars')
                original_instruction = session.chat_history.get_item_deep_copy(0).content if injection_chars is not None else None
                callback_handler.on_task_reset(callback_args)
                if original_instruction is not None:
                    enforce_injection_budget(session, original_instruction, int(injection_chars))
            logger.info(f"Sample {sample_index} start.")
            while session.sample_status == SampleStatus.RUNNING:
                if callback_args.session_controller.should_agent_inference:
                    agent.inference(session)
                    reason = invalid_session_reason(session, evaluated=False)
                    if reason:
                        raise RuntimeError(reason)
                    callback_handler.on_agent_inference(callback_args)
                if callback_args.session_controller.should_task_interact:
                    task.interact(session)
                    reason = invalid_session_reason(session, evaluated=False)
                    if reason:
                        raise RuntimeError(reason)
                    callback_handler.on_task_interact(callback_args)
            if callback_args.session_controller.should_task_complete:
                task.complete(session)
            reason = invalid_session_reason(session)
            if reason:
                raise RuntimeError(reason)
            # Prequential protocol: the evaluator runs before any learning callback.
            callback_handler.on_task_complete(callback_args)
            for model in config_utility.language_model_dict.values():
                if getattr(model, "audit_failure", None):
                    raise RuntimeError(f"Auxiliary inference failed: {model.audit_failure}")
            callback_handler.on_state_save(callback_args)
            new_sessions = session_list + [session]
            checkpoint.commit([item.model_dump() for item in new_sessions])
            session_list = new_sessions
            status("running")
            logger.info(f"Sample {sample_index} end. Session status: {session.sample_status}. Evaluation outcome: {session.evaluation_record.outcome}.")
        validate_prefix(session_list, assignment_config.sample_order, task.task_name)
        partials = [SessionMetricCalculationPartial(sample_index=session.sample_index,
                    evaluation_record=session.evaluation_record, sample_status=session.sample_status)
                    for session in session_list]
        metric = task.calculate_metric(partials)
        atomic_json(Path(path_config.metric_output_path), metric)

    except BaseException as error:
        execution_error = error
        if isinstance(error, UncommittedInferenceError):
            atomic_json(output / "resume_rejection.json", {"invalid_reasons": [str(error)]})
        if had_runtime_manifest and not identity_verified:
            # A refused invocation must not invalidate an existing completed result.
            atomic_json(output / "resume_rejection.json", {"invalid_reasons": [str(error)]})
        else:
            status("interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed", [str(error)])
        raise
    finally:
        if task is not None:
            try:
                task.release()
            except BaseException as cleanup_error:
                reason = f"Task cleanup failed: {cleanup_error}"
                reasons = ([str(execution_error)] if execution_error is not None else []) + [reason]
                if had_runtime_manifest and not identity_verified:
                    atomic_json(output / "resume_rejection.json", {"invalid_reasons": reasons})
                else:
                    status("failed", reasons)
                if execution_error is not None:
                    execution_error.add_note(reason)
                else:
                    raise
    status("completed")
    logger.info(f"Experiment end. Metric: {metric}. Total sample count: {expected_count}.")


if __name__ == "__main__":
    main()
