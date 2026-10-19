# Benchmark Data

The processed evaluation splits and prompt protocols required by all release
configs are included in this repository. Do not regenerate, reorder, or edit
these files when reproducing reported results.

The benchmark files use a stable public layout:

```text
lifelongagent_bench/
|-- db/entry_dict.json
|-- kg/entry_dict.json
|-- kg/ontology/
`-- os/entry_dict.json
```

Run the integrity check from `LifelongAgentBench/` after cloning:

```bash
python scripts/verify_data.py
```

`manifest.json` fixes the protocol version, sample counts, round limits, and
SHA-256 checksum of every evaluation input. DBBench and OS Interaction create
isolated Docker containers while running. Knowledge Graph additionally needs a
Freebase-compatible SPARQL service; the ontology files here are client-side
schema metadata, not the Freebase database itself.
