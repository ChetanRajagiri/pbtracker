import json
import os

def check_graph():
    intermediate_dir = "/Users/chetanrajagiri/Desktop/pbtracker/.understand-anything/intermediate"
    assembled_path = os.path.join(intermediate_dir, "assembled-graph.json")
    scan_result_path = os.path.join(intermediate_dir, "scan-result.json")
    
    with open(assembled_path, "r") as f:
        assembled = json.load(f)
        
    with open(scan_result_path, "r") as f:
        scan_result = json.load(f)
        
    nodes = assembled.get("nodes", [])
    edges = assembled.get("edges", [])
    
    print("=== Nodes Analysis ===")
    print(f"Total Nodes: {len(nodes)}")
    
    node_ids = set()
    node_types = set()
    node_complexities = set()
    missing_fields = {}
    empty_fields = {}
    
    required_node_fields = ["id", "type", "name", "filePath", "summary", "tags", "complexity"]
    
    for n in nodes:
        nid = n.get("id")
        if not nid:
            print("Node without ID:", n)
            continue
        if nid in node_ids:
            print("Duplicate node ID in assembled graph:", nid)
        node_ids.add(nid)
        
        # Check type
        ntype = n.get("type")
        node_types.add(ntype)
        
        # Check complexity
        ncomp = n.get("complexity")
        node_complexities.add(ncomp)
        
        # Check missing or empty fields
        for field in required_node_fields:
            if field not in n:
                missing_fields.setdefault(field, []).append(nid)
            else:
                val = n[field]
                if val is None or val == "" or (isinstance(val, list) and len(val) == 0):
                    # Empty list for tags is fine, but let's check
                    if field != "tags":
                        empty_fields.setdefault(field, []).append(nid)
                        
    print("Node Types found:", node_types)
    print("Node Complexities found:", node_complexities)
    print("Missing fields in nodes:")
    for field, nids in missing_fields.items():
        print(f"  - {field}: {len(nids)} nodes (e.g. {nids[:3]})")
    print("Empty/Null fields in nodes:")
    for field, nids in empty_fields.items():
        print(f"  - {field}: {len(nids)} nodes (e.g. {nids[:3]})")
        
    print("\n=== Edges Analysis ===")
    print(f"Total Edges: {len(edges)}")
    
    required_edge_fields = ["source", "target", "type", "direction"]
    missing_edge_fields = {}
    dangling_edges = []
    
    for idx, e in enumerate(edges):
        for field in required_edge_fields:
            if field not in e:
                missing_edge_fields.setdefault(field, []).append(idx)
        
        src = e.get("source")
        tgt = e.get("target")
        
        if src and src not in node_ids:
            dangling_edges.append((idx, "source", src, tgt))
        if tgt and tgt not in node_ids:
            dangling_edges.append((idx, "target", src, tgt))
            
    print("Missing fields in edges:")
    for field, idxs in missing_edge_fields.items():
        print(f"  - {field}: {len(idxs)} edges (e.g. {idxs[:3]})")
    print(f"Dangling edges found: {len(dangling_edges)}")
    for idx, role, src, tgt in dangling_edges[:5]:
        print(f"  - Edge {idx}: {role} '{src if role=='source' else tgt}' does not exist (Edge: {src} -> {tgt})")
        
    # Check for duplicate edges
    edge_tuples = []
    duplicate_edges = []
    for e in edges:
        etuple = (e.get("source"), e.get("target"), e.get("type"))
        if etuple in edge_tuples:
            duplicate_edges.append(etuple)
        edge_tuples.append(etuple)
    print(f"Duplicate edges in assembled graph: {len(duplicate_edges)}")
    for de in duplicate_edges[:5]:
        print(f"  - {de}")
        
    # Analyze relationship between batch files and assembled graph
    print("\n=== Batch Files Comparison ===")
    batch_files = [f"batch-{i}.json" for i in range(1, 6)]
    for bfile in batch_files:
        bpath = os.path.join(intermediate_dir, bfile)
        if not os.path.exists(bpath):
            continue
        with open(bpath, "r") as f:
            batch = json.load(f)
        bnodes = batch.get("nodes", [])
        bedges = batch.get("edges", [])
        
        # Check for unknown / null values in the batch itself
        b_unknown_types = [n["id"] for n in bnodes if n.get("type") in [None, "unknown", ""]]
        b_unknown_complexities = [n["id"] for n in bnodes if n.get("complexity") in [None, "unknown", ""]]
        
        print(f"{bfile}: {len(bnodes)} nodes, {len(bedges)} edges")
        if b_unknown_types:
            print(f"  - Unknown types: {b_unknown_types}")
        if b_unknown_complexities:
            print(f"  - Unknown complexities: {b_unknown_complexities}")
            
    # Check import map details
    print("\n=== Import Map Detailed Analysis ===")
    import_map = scan_result.get("importMap", {})
    all_imports_count = sum(len(v) for v in import_map.values())
    print(f"Total entries in importMap: {len(import_map)} files, total target imports: {all_imports_count}")
    
    missing_imports = []
    for src_file, imports in import_map.items():
        src_id = f"file:{src_file}"
        for imp_file in imports:
            tgt_id = f"file:{imp_file}"
            
            # Let's see if this edge exists in edges
            found = False
            for e in edges:
                if e.get("source") == src_id and e.get("target") == tgt_id and e.get("type") == "imports":
                    found = True
                    break
            if not found:
                missing_imports.append((src_id, tgt_id))
                
    print(f"Missing import edges: {len(missing_imports)}")
    for src, tgt in missing_imports:
        print(f"  - {src} -> {tgt}")
        
if __name__ == "__main__":
    check_graph()
