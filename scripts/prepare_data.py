import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from core.documents import create_document, process_code, process_md


TELEMINT_README = """
# Telemint
This is the smart contract that Telegram intends to use in order to put some of its best usernames up for auction. The blockchain network for this smart contract is The Open Network (https://ton.org).

Anyone who finds serious security vulnerabilities in this smart contract prior to the auction launch will be rewarded.

## Description
There are two smart contracts in the repository: NftCollection and NftItem.

NftCollection source files: [nft-collection.fc](func/nft-collection.fc), [common.fc](func/common.fc) [stdlib.fc](func/stdlib.fc).

NftItem source files: [nft-item.fc](func/nft-item.fc), [common.fc](func/common.fc) [stdlib.fc](func/stdlib.fc).

One may also look at the [tlb decription](telemint.tlb) of internal messages and smart contract data.

There are also two additional smart contracts in the repository: NftCollectionNoDns and NftItemNoDns. They do not support DNS and allow to set additional restrictions on first bid.

NftCollectionNoDns source files: [nft-collection-no-dns.fc](func/nft-collection-no-dns.fc), [common.fc](func/common.fc) [stdlib.fc](func/stdlib.fc).

NftItemNoDns source files: [nft-item-no-dns.fc](func/nft-item-no-dns.fc), [common.fc](func/common.fc) [stdlib.fc](func/stdlib.fc).

### NftCollection

#### Internal messages
The first bidder receives a signed query from the server and sends it to NftCollection with the first bid attached.
```
// Create an NftItem and start an auction. Signed by auction's private key. Acts as a first bid in the auction.
telemint_unsigned_deploy$_ subwallet_id:uint32 valid_since:uint32 valid_till:uint32 token_name:TelemintText
  content:^Cell auction_config:^TeleitemAuctionConfig royalty_params:(Maybe ^NftRoyaltyParams) = TelemintUnsignedDeploy;
telemint_msg_deploy#4637289a  sig:bits512 msg:TelemintUnsignedDeploy = TelemintMsg;
```

The NftCollection interface is also supported.

#### External messages
The smart contract will accept the first external message to simplify the initialization of the smart contract.

### NftItem

#### Internal messages
The first bid is made through NftCollection, which will generate the following message.
```
// Create NftItem and start an auction. Accepted only from NftCollection.
teleitem_msg_deploy#299a3e15 sender_address:MsgAddressInt bid:Grams token_info:^TelemintTokenInfo nft_content:^Cell
  auction_config:^TeleitemAuctionConfig royalty_params:^NftRoyaltyParams = TeleitemMsg;
```

All following bids are simple transfers.

The owner of an NftItem may start a new auction.

```
// Start new auction. Accepted only from the owner.
teleitem_msg_start_auction#487a8e81 query_id:int64 auction_config:^TeleitemAuctionConfig = TeleitemMsg;

// Cancel auction auction. Accepted only from the owner. Forbidden if there are some active bids
teleitem_msg_cancel_auction#371638ae query_id:int64 = TeleitemMsg;
```

The NftItem interface is also supported, including transfer messages.

#### External messages
To finish a completed auction, one may send an empty message.
""".strip()


def dump_json_docs(documents: Iterable[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as out_file:
        for doc in documents:
            out_file.write(doc.model_dump_json() + "\n")


def load_example_mappings(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Examples mapping file is missing: {path}. Create it before running data prep."
        )

    data = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(data, list):
        raise ValueError("Examples mapping json must be a list of mapping objects")

    required_keys = {"examples_path", "concept", "lang", "labels"}
    mappings: List[Dict[str, Any]] = []

    for index, raw_mapping in enumerate(data):
        if not isinstance(raw_mapping, dict):
            raise ValueError(f"Examples mapping entry at index {index} must be an object")

        missing = [key for key in required_keys if key not in raw_mapping]
        if missing:
            raise ValueError(
                f"Examples mapping entry at index {index} is missing required keys: {', '.join(missing)}"
            )

        mapping: Dict[str, Any] = {}
        for key in ("examples_path", "concept", "lang", "doc_text", "doc_text_key"):
            value = raw_mapping.get(key)
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError(
                    f"Examples mapping entry at index {index} key '{key}' must be a string"
                )
            mapping[key] = value

        labels = raw_mapping.get("labels")
        if not isinstance(labels, dict):
            raise ValueError(
                f"Examples mapping entry at index {index} key 'labels' must be an object"
            )

        parsed_labels: Dict[str, str] = {}
        for label_key, label_value in labels.items():
            if not isinstance(label_key, str) or not isinstance(label_value, str):
                raise ValueError(
                    f"Examples mapping entry at index {index} labels must contain string keys and values"
                )
            parsed_labels[label_key] = label_value.strip()

        mapping["labels"] = parsed_labels

        mappings.append(mapping)

    return mappings


def read_md_file(
    root_path: Path,
    md_path: Path,
    custom_crumbs: Optional[List[str]] = None,
) -> Tuple[List[Document], Dict[str, Document]]:
    doc_rel_path = str(md_path.relative_to(root_path))
    md_text = md_path.read_text(encoding="utf8")
    return process_md(doc_rel_path, md_text, custom_crumbs=custom_crumbs)


def resolve_document_references(docs: Sequence[Document]) -> None:
    location_map: Dict[str, List[str]] = {}

    for doc in docs:
        doc_id = doc.id
        url_from = doc.metadata.get("url_from")
        if not doc_id or not isinstance(url_from, str):
            continue
        if url_from in location_map:
            location_map[url_from].append(doc_id)
        else:
            location_map[url_from] = [doc_id]

    for doc in docs:
        references = doc.metadata.get("references", [])
        if not isinstance(references, list):
            continue
        ref_ids: List[str] = []
        for ref in references:
            if isinstance(ref, str) and ref in location_map:
                ref_ids.extend(location_map[ref])
        if ref_ids:
            doc.metadata["references"] = ref_ids.copy()


def add_examples(
    root_path: Path,
    snippets: Dict[str, Document],
    labels: Dict[str, str],
    examples_path: str,
    concept: str,
    lang: str,
    doc_text: str = "",
) -> List[Document]:
    code_refs: List[str] = []
    code_docs: List[Document] = []
    doc_chunks = [doc_text] if doc_text else []

    parsed_path = Path(examples_path)
    if not parsed_path.is_absolute():
        parsed_path = root_path / parsed_path

    top_level_doc = create_document(
        concept,
        doc_chunks,
        str(parsed_path.relative_to(root_path)),
        references=[],
        ref_snippets=[],
        crumbs=[concept],
        snippets=snippets,
        children_nodes=code_refs,
    )

    for rel_group_path in sorted(labels.keys()):
        doc = (parsed_path / rel_group_path).resolve()
        try:
            doc.relative_to(parsed_path.resolve())
        except ValueError as ex:
            raise ValueError(
                f"Invalid example path '{rel_group_path}' for base '{examples_path}'"
            ) from ex

        if not doc.is_file():
            raise ValueError(
                f"Example file from mapping does not exist: {doc.relative_to(root_path)}"
            )

        content = doc.read_text(encoding="utf8")
        rel_path = str(doc.relative_to(root_path))
        file_label = labels.get(rel_group_path)

        if file_label is None or not file_label.strip():
            raise ValueError(
                f"Missing label for example file: {rel_path}. Add it to examples mapping labels."
            )

        if file_label.lower() == "skip":
            continue

        source_doc = process_code(content, lang, concept, file_label)
        if source_doc.id is None:
            continue
        snippets[source_doc.id] = source_doc

        mention_doc = create_document(
            title=concept,
            doc_chunks=[],
            path=str(doc.relative_to(root_path)),
            references=[],
            ref_snippets=[{"id": source_doc.id, "pos": 0}],
            crumbs=[concept, file_label],
            snippets=snippets,
        )

        if mention_doc.id is None:
            continue
        code_docs.append(mention_doc)
        code_refs.append(mention_doc.id)

    return [top_level_doc, *code_docs]


def _fix_instruction_paragraphs(paragraphs: List[str], instruction: str) -> List[str]:
    header_re = r"^(#+)"
    hashes_added = 0
    paragraph_count = 0
    new_paragraphs: List[str] = []

    for p in paragraphs:
        match = re.match(header_re, p)
        if not match:
            continue

        hashes = match.group()
        total_hashes = len(hashes)
        if hashes_added == 0 and total_hashes < 3:
            hashes_added = 3 - total_hashes

        if paragraph_count > 0:
            new_paragraphs.append(("#" * hashes_added) + p)
        else:
            new_paragraphs.append(f"### {instruction}\n instruction specification")
            hashes_added = hashes_added + 1

        paragraph_count += 1

    return new_paragraphs


def load_instruction_documents(
    instructions_file: Path,
) -> Tuple[List[Document], Dict[str, Document]]:
    instructions_data = json.loads(instructions_file.read_text(encoding="utf8"))
    instruction_docs: List[Document] = []
    all_snippets: Dict[str, Document] = {}

    for instruction_category, category_docs in instructions_data.items():
        for instruction, markdown_body in category_docs.items():
            paragraphs = markdown_body.split("\n\n")
            fixed_paragraphs = _fix_instruction_paragraphs(paragraphs, instruction)
            cat_title = f"{instruction_category} instructions"

            docs, snippets = process_md(
                "tvm-specification.json",
                "\n\n".join(fixed_paragraphs),
                custom_title=instruction_category,
                custom_crumbs=["TVM instrucitons", cat_title],
                skip_top=True,
            )
            instruction_docs.extend(docs)
            all_snippets.update(snippets)

    return instruction_docs, all_snippets


def build_data(
    root_path: Path, rag_data_path: Path, include_instructions: bool
) -> None:
    docs: List[Document] = []
    snippets: Dict[str, Document] = {}
    example_mappings = load_example_mappings(rag_data_path / "examples_mapping.json")

    doc_text_key_map = {
        "telemint_readme": TELEMINT_README,
    }

    docs_path = root_path / "docs-data"
    for file in sorted(docs_path.rglob("*.mdx")):
        crumbs = list(file.relative_to(docs_path).parts)[:-1]
        new_docs, new_snippets = read_md_file(root_path, file, custom_crumbs=crumbs)
        docs.extend(new_docs)
        snippets.update(new_snippets)

    teps_path = root_path / "TEPs" / "text"
    for file in sorted(teps_path.rglob("*.md")):
        new_docs, new_snippets = read_md_file(root_path, file)
        docs.extend(new_docs)
        snippets.update(new_snippets)

    resolve_document_references(docs)

    src_examples: List[Document] = []
    for mapping in example_mappings:
        mapping_labels = mapping["labels"]
        doc_text = mapping.get("doc_text", "")
        doc_text_key = mapping.get("doc_text_key")
        if doc_text_key:
            if doc_text_key not in doc_text_key_map:
                raise ValueError(
                    f"Unknown doc_text_key '{doc_text_key}' in examples mapping"
                )
            doc_text = doc_text_key_map[doc_text_key]

        src_examples.extend(
            add_examples(
                root_path,
                snippets,
                mapping_labels,
                mapping["examples_path"],
                mapping["concept"],
                mapping["lang"],
                doc_text=doc_text,
            )
        )

    instruction_docs: List[Document] = []
    instructions_file = rag_data_path / "instructions_desc.json"
    if include_instructions and instructions_file.exists():
        instruction_docs, instruction_snippets = load_instruction_documents(
            instructions_file
        )
        snippets.update(instruction_snippets)
        # dump_json_docs(instruction_docs, rag_data_path / "instructions_documents.jsonl")

    latest_docs = [*docs, *src_examples, *instruction_docs]
    latest_snippets = list(snippets.values())

    dump_json_docs(latest_docs, rag_data_path / "docs.jsonl")
    dump_json_docs(latest_snippets, rag_data_path / "latest_snippets.jsonl")

    print(f"Base docs: {len(docs)}")
    print(f"Source example docs: {len(src_examples)}")
    print(f"Instruction docs: {len(instruction_docs)}")
    print(f"Total docs (latest_docs.jsonl): {len(latest_docs)}")
    print(f"Total snippets (latest_snippets.jsonl): {len(latest_snippets)}")
    print(f"Saved to: {rag_data_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build jsonl data dumps for RAG index setup"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project root path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output rag-data directory (default: <root>/rag-data)",
    )
    parser.add_argument(
        "--skip-instructions",
        action="store_true",
        help="Skip optional TVM instruction processing",
    )
    args = parser.parse_args()

    root_path = args.root.resolve()
    rag_data_path = args.output.resolve() if args.output else root_path / "rag-data"

    build_data(
        root_path=root_path,
        rag_data_path=rag_data_path,
        include_instructions=not args.skip_instructions,
    )


if __name__ == "__main__":
    main()
