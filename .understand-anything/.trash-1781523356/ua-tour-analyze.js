const fs = require('fs');
const path = require('path');

const rootDir = '/Users/chetanrajagiri/Desktop/pbtracker';
const inputPath = path.join(rootDir, '.understand-anything/tmp/ua-tour-input.json');
const outputPath = path.join(rootDir, '.understand-anything/tmp/ua-tour-results.json');

try {
  if (!fs.existsSync(inputPath)) {
    // If the input JSON doesn't exist yet, we try to create it from intermediate files directly
    console.log('Input file not found at ' + inputPath + ', trying to create it directly...');
    const graphPath = path.join(rootDir, '.understand-anything/intermediate/assembled-graph.json');
    const layersPath = path.join(rootDir, '.understand-anything/intermediate/layers.json');
    
    if (fs.existsSync(graphPath) && fs.existsSync(layersPath)) {
      const graph = JSON.parse(fs.readFileSync(graphPath, 'utf8'));
      const layers = JSON.parse(fs.readFileSync(layersPath, 'utf8'));
      fs.writeFileSync(inputPath, JSON.stringify({ graph, layers }, null, 2));
      console.log('Successfully created input file.');
    } else {
      throw new Error('Assembled graph or layers file missing!');
    }
  }

  const inputData = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const graph = inputData.graph;
  const layers = inputData.layers;

  const nodes = graph.nodes || [];
  const edges = graph.edges || [];

  // 1. Analyze node types
  const nodeTypes = {};
  nodes.forEach(n => {
    nodeTypes[n.type] = (nodeTypes[n.type] || 0) + 1;
  });

  // 2. Compute degrees
  const inDegree = {};
  const outDegree = {};
  
  nodes.forEach(n => {
    inDegree[n.id] = 0;
    outDegree[n.id] = 0;
  });

  edges.forEach(e => {
    if (inDegree[e.target] !== undefined) {
      inDegree[e.target]++;
    }
    if (outDegree[e.source] !== undefined) {
      outDegree[e.source]++;
    }
  });

  // 3. Find top central files (by in-degree and out-degree)
  const fileNodes = nodes.filter(n => n.type === 'file' || n.type === 'document' || n.type === 'config');
  
  const mostImported = [...fileNodes]
    .map(n => ({ id: n.id, in: inDegree[n.id] || 0 }))
    .sort((a, b) => b.in - a.in)
    .slice(0, 5);

  const mostImporting = [...fileNodes]
    .map(n => ({ id: n.id, out: outDegree[n.id] || 0 }))
    .sort((a, b) => b.out - a.out)
    .slice(0, 5);

  // 4. Map layers and their nodes
  const layerStats = layers.map(layer => {
    const layerNodes = layer.nodeIds || [];
    const internalEdges = [];
    const externalIncoming = [];
    const externalOutgoing = [];

    edges.forEach(e => {
      const srcInLayer = layerNodes.includes(e.source);
      const tgtInLayer = layerNodes.includes(e.target);

      if (srcInLayer && tgtInLayer) {
        internalEdges.push(e);
      } else if (!srcInLayer && tgtInLayer) {
        externalIncoming.push(e);
      } else if (srcInLayer && !tgtInLayer) {
        externalOutgoing.push(e);
      }
    });

    return {
      layerId: layer.id,
      layerName: layer.name,
      nodeCount: layerNodes.length,
      internalEdgesCount: internalEdges.length,
      incomingEdgesCount: externalIncoming.length,
      outgoingEdgesCount: externalOutgoing.length
    };
  });

  const results = {
    analyzedAt: new Date().toISOString(),
    stats: {
      totalNodes: nodes.length,
      totalEdges: edges.length,
      nodeTypes
    },
    centrality: {
      mostImported,
      mostImporting
    },
    layers: layerStats
  };

  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
  console.log('Successfully wrote topology analysis results to ' + outputPath);
} catch (error) {
  console.error('Error in graph analysis script:', error);
  process.exit(1);
}
