#!/usr/bin/env python3
"""Build the exact AutoDrive Egypt canvas from the live Langflow registry."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from lfx.graph.flow_builder import add_component, add_connection, configure_component, empty_flow, layout_flow
from lfx.mcp.client import LangflowClient
from lfx.mcp.registry import load_registry

FLOW_NAME = "Car Dealership AI Agent — Gemini 3.5 Flash-Lite"

REQUIRED = [
    "ChatInput",
    "InputGuardrails",
    "GeminiCarSalesAgent",
    "SearchUsedCars",
    "GetCarDetails",
    "CompareCars",
    "DealershipKnowledgeRAG",
    "CreateTestDrive",
    "CancelTestDrive",
    "CreateSalesLead",
    "VehicleResearchAgent",
    "UnifiedWebSearch",
    "URLComponent",
    "OutputGuardrails",
    "ChatOutput",
]

SALES_TOOLS = [
    "SearchUsedCars",
    "GetCarDetails",
    "CompareCars",
    "DealershipKnowledgeRAG",
    "CreateTestDrive",
    "CancelTestDrive",
    "CreateSalesLead",
    "VehicleResearchAgent",
]

RESEARCH_WEB_TOOLS = ["UnifiedWebSearch", "URLComponent"]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=os.getenv("LANGFLOW_SERVER_URL", "http://localhost:7860"))
    parser.add_argument("--api-key", default=os.getenv("LANGFLOW_API_KEY", ""))
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "flow" / "Car_Dealership_Agent.flow.json"),
    )
    args = parser.parse_args()

    client = LangflowClient(server_url=args.server, api_key=args.api_key or None)
    if not args.api_key:
        client.api_key = None

    try:
        if not args.api_key:
            login = await client.get("/auto_login")
            if not isinstance(login, dict) or not login.get("access_token"):
                raise RuntimeError(f"Auto-login did not return access_token: {login!r}")
            client.access_token = login["access_token"]
            client.api_key = None
            print("[OK] Authenticated to Langflow using AUTO_LOGIN bearer token")

        registry = await load_registry(client)

        def resolve_registry_type(logical_name: str) -> str | None:
            if logical_name in registry:
                return logical_name
            for key, entry in registry.items():
                if f":{logical_name}@" in key or key.endswith(f":{logical_name}"):
                    return key
                if isinstance(entry, dict):
                    for field in ("name", "key", "type"):
                        if entry.get(field) == logical_name:
                            return key
                    template = entry.get("template", {})
                    if isinstance(template, dict) and template.get("_type") == logical_name:
                        return key
            return None

        resolved_types = {name: resolve_registry_type(name) for name in REQUIRED}
        missing = [name for name, actual in resolved_types.items() if actual is None]
        if missing:
            relevant = [
                key for key in registry
                if key.startswith("ext:") or "car" in key.lower() or "guard" in key.lower()
            ]
            raise RuntimeError(
                "Components still missing after resolving Langflow extension names: "
                + ", ".join(missing)
                + f". Relevant registry keys discovered: {relevant[:50]}"
            )

        print("[OK] Resolved component registry names:")
        for logical, actual in resolved_types.items():
            print(f"  {logical} -> {actual}")

        flow = empty_flow(
            name=FLOW_NAME,
            description=(
                "Egypt new+used sales orchestrator with hybrid memory, deterministic business gates, "
                "RAG, and a subordinate vehicle research agent."
            ),
        )
        ids: dict[str, str] = {}
        for component_type in REQUIRED:
            ids[component_type] = add_component(
                flow,
                resolved_types[component_type],
                registry,
            )["id"]

        configure_component(flow, ids["GeminiCarSalesAgent"], {
            "model": "gemini-3.5-flash-lite",
            "db_path": "/data/car_dealership.db",
            "memory_messages": 12,
            "max_iterations": 10,
        })
        configure_component(flow, ids["SearchUsedCars"], {"db_path": "/data/car_dealership.db", "limit": 5})
        configure_component(flow, ids["GetCarDetails"], {"db_path": "/data/car_dealership.db"})
        configure_component(flow, ids["CompareCars"], {"db_path": "/data/car_dealership.db"})
        configure_component(flow, ids["CreateTestDrive"], {"db_path": "/data/car_dealership.db"})
        configure_component(flow, ids["CancelTestDrive"], {"db_path": "/data/car_dealership.db"})
        configure_component(flow, ids["CreateSalesLead"], {"db_path": "/data/car_dealership.db"})
        configure_component(flow, ids["VehicleResearchAgent"], {
            "db_path": "/data/car_dealership.db",
            "model_name": "gemini-3.5-flash-lite",
            "max_iterations": 8,
        })
        configure_component(flow, ids["UnifiedWebSearch"], {
            "search_mode": "Web",
            "max_results": 3,
            "max_content_length": 1800,
        })
        configure_component(flow, ids["URLComponent"], {
            "max_depth": 1,
            "format": "Text",
            "timeout": 12,
        })
        configure_component(flow, ids["DealershipKnowledgeRAG"], {
            "db_path": "/data/car_dealership.db",
            "knowledge_dir": "/app/knowledge",
            "chroma_dir": "/data/chroma",
        })

        # Customer-facing path.
        add_connection(flow, ids["ChatInput"], "message", ids["InputGuardrails"], "input_value", registry=registry)
        add_connection(flow, ids["InputGuardrails"], "message", ids["GeminiCarSalesAgent"], "input_value", registry=registry)
        add_connection(flow, ids["GeminiCarSalesAgent"], "response", ids["OutputGuardrails"], "input_value", registry=registry)
        add_connection(flow, ids["OutputGuardrails"], "message", ids["ChatOutput"], "input_value", registry=registry)

        # The Sales Orchestrator can call only dealership/business tools plus the subordinate research agent.
        for tool_type in SALES_TOOLS:
            add_connection(
                flow,
                ids[tool_type],
                "component_as_tool",
                ids["GeminiCarSalesAgent"],
                "tools",
                registry=registry,
            )

        # Generic web tools are isolated behind VehicleResearchAgent; SalesAgent cannot bypass it.
        for tool_type in RESEARCH_WEB_TOOLS:
            add_connection(
                flow,
                ids[tool_type],
                "component_as_tool",
                ids["VehicleResearchAgent"],
                "tools",
                registry=registry,
            )

        layout_flow(flow)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(flow, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Exported exact flow JSON: {out}")
        print(f"[OK] Nodes={len(flow['data']['nodes'])}, Edges={len(flow['data']['edges'])}")

        if not args.no_install:
            current = await client.get("/flows/")
            current_flows = current if isinstance(current, list) else current.get("items", current.get("flows", []))
            matches = [item for item in current_flows if item.get("name") == FLOW_NAME]
            if len(matches) > 1:
                raise RuntimeError(
                    f"Found {len(matches)} existing flows named {FLOW_NAME!r}; refusing ambiguous update."
                )
            if matches:
                flow_id = matches[0]["id"]
                updated = await client.patch(
                    f"/flows/{flow_id}",
                    json_data={"name": flow["name"], "description": flow["description"], "data": flow["data"]},
                )
                print(f"[OK] Updated existing Langflow flow: id={updated.get('id')} name={updated.get('name')}")
            else:
                created = await client.post("/flows/", json_data=flow)
                print(f"[OK] Installed new Langflow flow: id={created.get('id')} name={created.get('name')}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
