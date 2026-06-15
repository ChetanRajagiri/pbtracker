const fs = require('fs');
const path = require('path');

const root = '/Users/chetanrajagiri/Desktop/pbtracker';
const interDir = path.join(root, '.understand-anything', 'intermediate');
const tmpDir = path.join(root, '.understand-anything', 'tmp');

try {
  const assembledGraph = JSON.parse(fs.readFileSync(path.join(interDir, 'assembled-graph.json'), 'utf8'));
  const layers = JSON.parse(fs.readFileSync(path.join(interDir, 'layers.json'), 'utf8'));
  const tour = JSON.parse(fs.readFileSync(path.join(interDir, 'tour.json'), 'utf8'));

  const fullGraph = {
    version: "1.0.0",
    project: {
      name: "pbtracker",
      languages: ["html", "json", "markdown", "python", "toml", "yaml"],
      frameworks: ["OpenCV", "Streamlit", "PyTorch", "Ultralytics", "Pandas", "Matplotlib", "Plotly"],
      description: "A computer vision pipeline for tracking players and the ball in pickleball match footage. Built with YOLO object detection, OpenCV geometric filtering, and deep learning-based person re-identification.",
      analyzedAt: new Date().toISOString(),
      gitCommitHash: "27ec0c959adc1dfa83ce8ff0278adeb0905216e5"
    },
    nodes: assembledGraph.nodes || [],
    edges: assembledGraph.edges || [],
    layers: layers,
    tour: tour
  };

  // Run the validator logic
  const issues = [], warnings = [];
  if (!Array.isArray(fullGraph.nodes)) { issues.push('graph.nodes is missing or not an array'); fullGraph.nodes = []; }
  if (!Array.isArray(fullGraph.edges)) { issues.push('graph.edges is missing or not an array'); fullGraph.edges = []; }
  
  const nodeIds = new Set();
  const seen = new Map();
  fullGraph.nodes.forEach((n, i) => {
    if (!n.id) { issues.push(`Node[${i}] missing id`); return; }
    if (!n.type) issues.push(`Node[${i}] '${n.id}' missing type`);
    if (!n.name) issues.push(`Node[${i}] '${n.id}' missing name`);
    if (!n.summary) issues.push(`Node[${i}] '${n.id}' missing summary`);
    if (!n.tags || !n.tags.length) issues.push(`Node[${i}] '${n.id}' missing tags`);
    if (seen.has(n.id)) issues.push(`Duplicate node ID '${n.id}' at indices ${seen.get(n.id)} and ${i}`);
    else seen.set(n.id, i);
    nodeIds.add(n.id);
  });

  fullGraph.edges.forEach((e, i) => {
    if (!nodeIds.has(e.source)) issues.push(`Edge[${i}] source '${e.source}' not found`);
    if (!nodeIds.has(e.target)) issues.push(`Edge[${i}] target '${e.target}' not found`);
  });

  const fileLevelTypes = new Set(['file', 'config', 'document', 'service', 'pipeline', 'table', 'schema', 'resource', 'endpoint']);
  const fileNodes = fullGraph.nodes.filter(n => fileLevelTypes.has(n.type)).map(n => n.id);
  const assigned = new Map();
  if (!Array.isArray(fullGraph.layers)) { if (fullGraph.layers) warnings.push('graph.layers is not an array'); fullGraph.layers = []; }
  if (!Array.isArray(fullGraph.tour)) { if (fullGraph.tour) warnings.push('graph.tour is not an array'); fullGraph.tour = []; }
  
  fullGraph.layers.forEach(layer => {
    (layer.nodeIds || []).forEach(id => {
      if (!nodeIds.has(id)) issues.push(`Layer '${layer.id}' refs missing node '${id}'`);
      if (assigned.has(id)) issues.push(`Node '${id}' appears in multiple layers`);
      assigned.set(id, layer.id);
    });
  });

  fileNodes.forEach(id => {
    if (!assigned.has(id)) {
      // Auto-assign any unassigned file node to the closest/most sensible layer.
      if (id.startsWith('file:scratch/')) {
        const diagLayer = fullGraph.layers.find(l => l.id === 'layer:diagnostics-scratch');
        if (diagLayer) {
          diagLayer.nodeIds.push(id);
          assigned.set(id, 'layer:diagnostics-scratch');
        }
      } else if (id.startsWith('config:') || id.startsWith('document:') || id.startsWith('file:.understand-anything/')) {
        const infraLayer = fullGraph.layers.find(l => l.id === 'layer:infrastructure');
        if (infraLayer) {
          infraLayer.nodeIds.push(id);
          assigned.set(id, 'layer:infrastructure');
        }
      } else {
        issues.push(`File node '${id}' not in any layer`);
      }
    }
  });

  fullGraph.tour.forEach((step, i) => {
    (step.nodeIds || []).forEach(id => {
      if (!nodeIds.has(id)) issues.push(`Tour step[${i}] refs missing node '${id}'`);
    });
  });

  const withEdges = new Set([
    ...fullGraph.edges.map(e => e.source),
    ...fullGraph.edges.map(e => e.target)
  ]);
  fullGraph.nodes.forEach(n => {
    if (!withEdges.has(n.id)) warnings.push(`Node '${n.id}' has no edges (orphan)`);
  });

  const stats = {
    totalNodes: fullGraph.nodes.length,
    totalEdges: fullGraph.edges.length,
    totalLayers: fullGraph.layers.length,
    tourSteps: fullGraph.tour.length,
    nodeTypes: fullGraph.nodes.reduce((a, n) => { a[n.type] = (a[n.type]||0)+1; return a; }, {}),
    edgeTypes: fullGraph.edges.reduce((a, e) => { a[e.type] = (a[e.type]||0)+1; return a; }, {})
  };

  fs.writeFileSync(path.join(interDir, 'assembled-graph.json'), JSON.stringify(fullGraph, null, 2));
  fs.writeFileSync(path.join(interDir, 'review.json'), JSON.stringify({ issues, warnings, stats }, null, 2));
  console.log(`Assembly and validation complete. Issues: ${issues.length}, Warnings: ${warnings.length}`);
  if (issues.length > 0) {
    console.error('Validation issues found:', issues);
  }
  process.exit(0);
} catch (err) {
  console.error(err);
  process.exit(1);
}
