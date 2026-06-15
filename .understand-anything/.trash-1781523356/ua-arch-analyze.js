const fs = require('fs');
const path = require('path');

const projectRoot = '/Users/chetanrajagiri/Desktop/pbtracker';
const assembledGraphPath = path.join(projectRoot, '.understand-anything/intermediate/assembled-graph.json');
const inputJsonPath = path.join(projectRoot, '.understand-anything/tmp/ua-arch-input.json');
const resultsJsonPath = path.join(projectRoot, '.understand-anything/tmp/ua-arch-results.json');
const layersJsonPath = path.join(projectRoot, '.understand-anything/intermediate/layers.json');

try {
  // 1. Read the assembled graph
  console.log('Reading assembled graph...');
  const graphContent = fs.readFileSync(assembledGraphPath, 'utf8');
  const graph = JSON.parse(graphContent);

  // 2. Filter the nodes to get all file-level nodes
  const fileLevelTypes = new Set(['file', 'config', 'document', 'service', 'pipeline', 'table', 'schema', 'resource', 'endpoint']);
  const fileNodes = graph.nodes.filter(n => fileLevelTypes.has(n.type));
  const fileNodeIds = new Set(fileNodes.map(n => n.id));

  console.log(`Found ${fileNodes.length} file-level nodes out of ${graph.nodes.length} total nodes.`);

  // Create a mapping of all node IDs to their parent file-level node ID
  const nodeIdToFileId = new Map();
  
  // First, map file-level nodes to themselves
  fileNodes.forEach(n => {
    nodeIdToFileId.set(n.id, n.id);
  });

  // Then map non-file-level nodes based on their filePath
  graph.nodes.forEach(n => {
    if (!fileLevelTypes.has(n.type) && n.filePath) {
      // Find the file node with matching filePath
      const correspondingFileNode = fileNodes.find(fn => fn.filePath === n.filePath);
      if (correspondingFileNode) {
        nodeIdToFileId.set(n.id, correspondingFileNode.id);
      }
    }
  });

  // 3. Map edges to dependencies between file nodes
  const dependencies = [];
  const seenDependencies = new Set();

  graph.edges.forEach(e => {
    const sourceFileId = nodeIdToFileId.get(e.source);
    const targetFileId = nodeIdToFileId.get(e.target);

    if (sourceFileId && targetFileId && sourceFileId !== targetFileId) {
      const depKey = `${sourceFileId}->${targetFileId}`;
      if (!seenDependencies.has(depKey)) {
        seenDependencies.add(depKey);
        dependencies.push({
          source: sourceFileId,
          target: targetFileId,
          type: e.type
        });
      }
    }
  });

  console.log(`Mapped ${dependencies.length} unique file-level dependencies.`);

  // 4. Prepare the input JSON
  const inputData = {
    fileNodes: fileNodes.map(n => ({
      id: n.id,
      type: n.type,
      name: n.name,
      filePath: n.filePath,
      summary: n.summary,
      tags: n.tags || []
    })),
    dependencies: dependencies
  };

  // Ensure tmp directory exists
  const tmpDir = path.dirname(inputJsonPath);
  if (!fs.existsSync(tmpDir)) {
    fs.mkdirSync(tmpDir, { recursive: true });
  }

  fs.writeFileSync(inputJsonPath, JSON.stringify(inputData, null, 2), 'utf8');
  console.log(`Wrote input JSON to ${inputJsonPath}`);

  // 5. Group the files into logical layers
  const layers = [
    {
      id: 'layer:presentation',
      name: 'Presentation Layer',
      description: 'Handles dashboard UI rendering and interactive visualizations.',
      nodeIds: []
    },
    {
      id: 'layer:orchestration',
      name: 'Application Orchestrator',
      description: 'Acts as the execution entry point and orchestrates the video processing pipeline.',
      nodeIds: []
    },
    {
      id: 'layer:tracking-detection',
      name: 'Computer Vision Trackers',
      description: 'Implements player and ball tracking systems using YOLO object detection models.',
      nodeIds: []
    },
    {
      id: 'layer:court-geometry',
      name: 'Court Geometry & Mapping',
      description: 'Handles homography transformations, perspective projection, and court selector GUI.',
      nodeIds: []
    },
    {
      id: 'layer:officiating-telemetry',
      name: 'Officiating & Telemetry Engine',
      description: 'Processes kinematics, attributions, line calls, and HUD telemetry overlays.',
      nodeIds: []
    },
    {
      id: 'layer:reid-healing',
      name: 'Data Healing & Re-ID',
      description: 'Post-processes and stitches fragmented player tracklets using visual embedding models.',
      nodeIds: []
    },
    {
      id: 'layer:diagnostics-scratch',
      name: 'Diagnostics & Scratchpad',
      description: 'Temporary scripts, one-off analyses, data checks, and frame annotation debug utilities.',
      nodeIds: []
    },
    {
      id: 'layer:infrastructure',
      name: 'Project Infrastructure & Config',
      description: 'Configuration files, ignore lists, dependency definitions, and project documentation.',
      nodeIds: []
    }
  ];

  const layerMap = new Map(layers.map(l => [l.id, l]));

  // Assign every file node to exactly one layer
  fileNodes.forEach(n => {
    const filePath = n.filePath;
    
    if (filePath.startsWith('scratch/') || filePath.endsWith('botsort_custom.yaml')) {
      layerMap.get('layer:diagnostics-scratch').nodeIds.push(n.id);
    } else if (filePath === 'app.py' || filePath === 'graphify.html') {
      layerMap.get('layer:presentation').nodeIds.push(n.id);
    } else if (filePath === 'main.py') {
      layerMap.get('layer:orchestration').nodeIds.push(n.id);
    } else if (filePath.startsWith('trackers/')) {
      layerMap.get('layer:tracking-detection').nodeIds.push(n.id);
    } else if (filePath.startsWith('court_line_detector/')) {
      layerMap.get('layer:court-geometry').nodeIds.push(n.id);
    } else if (filePath === 'utils/tracklet_merger.py' || filePath === 'utils/deep_reid_healer.py') {
      layerMap.get('layer:reid-healing').nodeIds.push(n.id);
    } else if (filePath === 'utils/generate_id_debug_frame.py' || filePath === 'utils/check_ids.py') {
      layerMap.get('layer:diagnostics-scratch').nodeIds.push(n.id);
    } else if (filePath.startsWith('utils/')) {
      layerMap.get('layer:officiating-telemetry').nodeIds.push(n.id);
    } else {
      // Configuration, documents, root config files
      layerMap.get('layer:infrastructure').nodeIds.push(n.id);
    }
  });

  // Filter out any empty layers
  const finalLayers = layers.filter(l => l.nodeIds.length > 0);

  // Validate that every file node is assigned to exactly one layer
  const assignedNodes = new Set();
  finalLayers.forEach(l => {
    l.nodeIds.forEach(id => {
      if (assignedNodes.has(id)) {
        throw new Error(`Node ${id} is assigned to multiple layers!`);
      }
      assignedNodes.add(id);
    });
  });

  fileNodes.forEach(n => {
    if (!assignedNodes.has(n.id)) {
      throw new Error(`Node ${n.id} is not assigned to any layer!`);
    }
  });

  console.log(`Validation successful: all ${fileNodes.length} nodes assigned to exactly one layer.`);

  // Write layers.json
  const intermediateDir = path.dirname(layersJsonPath);
  if (!fs.existsSync(intermediateDir)) {
    fs.mkdirSync(intermediateDir, { recursive: true });
  }
  fs.writeFileSync(layersJsonPath, JSON.stringify(finalLayers, null, 2), 'utf8');
  console.log(`Wrote layers JSON to ${layersJsonPath}`);

  // 6. Write results JSON
  const results = {
    analyzedAt: new Date().toISOString(),
    totalFiles: fileNodes.length,
    layers: finalLayers.map(l => ({
      id: l.id,
      name: l.name,
      nodeCount: l.nodeIds.length,
      nodeIds: l.nodeIds
    }))
  };
  fs.writeFileSync(resultsJsonPath, JSON.stringify(results, null, 2), 'utf8');
  console.log(`Wrote analysis results to ${resultsJsonPath}`);

} catch (err) {
  console.error('Error in analysis:', err);
  process.exit(1);
}
