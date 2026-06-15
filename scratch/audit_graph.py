import json
import os

def audit():
    intermediate_dir = "/Users/chetanrajagiri/Desktop/pbtracker/.understand-anything/intermediate"
    assembled_path = os.path.join(intermediate_dir, "assembled-graph.json")
    scan_result_path = os.path.join(intermediate_dir, "scan-result.json")
    
    with open(assembled_path, "r") as f:
        assembled = json.load(f)
        
    with open(scan_result_path, "r") as f:
        scan_result = json.load(f)
        
    # Map of assembled nodes
    assembled_nodes = {node["id"]: node for node in assembled.get("nodes", [])}
    # Set of assembled edges (as normalized tuples to handle uniqueness/direction check)
    assembled_edges = set()
    for edge in assembled.get("edges", []):
        src = edge["source"]
        tgt = edge["target"]
        etype = edge["type"]
        assembled_edges.add((src, tgt, etype))
        
    print(f"Assembled graph: {len(assembled_nodes)} nodes, {len(assembled_edges)} edges")
    
    # 1. Unknown types and complexities
    unknown_types = []
    unknown_complexities = []
    for nid, node in assembled_nodes.items():
        ntype = node.get("type", "unknown")
        if ntype == "unknown" or not ntype:
            unknown_types.append(node)
        complexity = node.get("complexity", "unknown")
        if complexity == "unknown" or not complexity:
            unknown_complexities.append(node)
            
    print(f"Unknown types count: {len(unknown_types)}")
    for u in unknown_types[:5]:
        print(f"  - Node ID: {u['id']}")
    print(f"Unknown complexities count: {len(unknown_complexities)}")
    for u in unknown_complexities[:5]:
        print(f"  - Node ID: {u['id']}")
        
    # 2. Check batch files for dropped nodes/edges
    batch_files = [f"batch-{i}.json" for i in range(1, 6)]
    dropped_nodes = []
    dropped_edges = []
    
    for bfile in batch_files:
        bpath = os.path.join(intermediate_dir, bfile)
        if not os.path.exists(bpath):
            print(f"Batch file not found: {bpath}")
            continue
        with open(bpath, "r") as f:
            batch = json.load(f)
        
        # Check nodes
        for node in batch.get("nodes", []):
            nid = node["id"]
            if nid not in assembled_nodes:
                dropped_nodes.append((bfile, node))
                
        # Check edges
        for edge in batch.get("edges", []):
            src = edge["source"]
            tgt = edge["target"]
            etype = edge["type"]
            if (src, tgt, etype) not in assembled_edges:
                dropped_edges.append((bfile, edge))
                
    print(f"Dropped nodes count: {len(dropped_nodes)}")
    for bfile, n in dropped_nodes[:5]:
        print(f"  - From {bfile}: {n['id']}")
    print(f"Dropped edges count: {len(dropped_edges)}")
    for bfile, e in dropped_edges[:5]:
        print(f"  - From {bfile}: {e['source']} -> {e['target']} ({e['type']})")
        
    # 3. Check cross-batch edge gaps using importMap
    import_map = scan_result.get("importMap", {})
    missing_import_edges = []
    for src_file, imports in import_map.items():
        src_id = f"file:{src_file}"
        for imp_file in imports:
            tgt_id = f"file:{imp_file}"
            # Verify if nodes exist
            if src_id not in assembled_nodes:
                print(f"Import map source node not in assembled: {src_id}")
                continue
            if tgt_id not in assembled_nodes:
                print(f"Import map target node not in assembled: {tgt_id}")
                continue
            
            # Check if edge exists
            if (src_id, tgt_id, "imports") not in assembled_edges:
                missing_import_edges.append((src_id, tgt_id))
                
    print(f"Missing import edges count: {len(missing_import_edges)}")
    for src_id, tgt_id in missing_import_edges:
        print(f"  - {src_id} -> {tgt_id}")
        
    return {
        "unknown_types": unknown_types,
        "unknown_complexities": unknown_complexities,
        "dropped_nodes": dropped_nodes,
        "dropped_edges": dropped_edges,
        "missing_import_edges": missing_import_edges
    }

if __name__ == "__main__":
    audit()
