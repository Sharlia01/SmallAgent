"""Reparse registered documents without deleting/reuploading their SQL records."""

import argparse
import json
from pathlib import Path

import xxhash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-name", required=True, help="Existing user's ES index")
    parser.add_argument("--file", type=Path, action="append", required=True)
    parser.add_argument("--apply", action="store_true", help="Reparse and replace; default only checks files and index")
    args = parser.parse_args()
    from service.core.rag.utils.es_conn import ESConnection
    connection = ESConnection()
    if not connection.es.indices.exists(index=args.index_name):
        parser.error("Target index does not exist")
    files = [path.resolve(strict=True) for path in args.file]
    if len({path.name for path in files}) != len(files):
        parser.error("Duplicate document filenames")
    # Validate every target before any replacement. No SQL mutation is needed.
    for path in files:
        if not path.is_file():
            parser.error(f"Not a file: {path}")
        doc_id = xxhash.xxh64(path.name.encode()).hexdigest()
        count = connection.es.count(index=args.index_name, query={"term": {"doc_id": doc_id}})["count"]
        if not count:
            parser.error(f"Document is not present in this index: {path.name}")
        print(json.dumps({"file": str(path), "existing_chunks": count, "apply": args.apply}, ensure_ascii=False), flush=True)
    if args.apply:
        from service.core.file_parse import execute_insert_process
        for path in files:
            result = execute_insert_process(str(path), path.name, args.index_name)
            print(json.dumps({"file": path.name, **result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
