from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Any, Optional


class GeneralInstanceFactory(BaseModel):
    module: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("parameters", mode="before")  # noqa
    @classmethod
    def _ensure_dict(cls, v: Optional[dict[str, Any]]) -> dict[str, Any]:
        # Follow the documentation, this function is defined as a class method
        # https://docs.pydantic.dev/2.10/concepts/validators/#using-the-decorator-pattern
        if v is None:
            return {}
        return v

    def create(self) -> Any:
        for parameter in self.parameters:
            try:
                self.parameters[parameter] = self.model_validate(
                    self.parameters[parameter]
                ).create()
            except ValidationError:
                pass
        module_path, separator, class_name = self.module.rpartition(".")
        if not separator:
            raise ValueError(f"Expected a fully qualified module path: {self.module}")
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)(**self.parameters)
