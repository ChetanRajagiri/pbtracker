const fs = require('fs');
const path = require('path');

const rootDir = '/Users/chetanrajagiri/Desktop/pbtracker';
const graphPath = path.join(rootDir, '.understand-anything/intermediate/assembled-graph.json');
const layersPath = path.join(rootDir, '.understand-anything/intermediate/layers.json');
const outputPath = path.join(rootDir, '.understand-anything/tmp/ua-tour-input.json');

try {
  const graph = JSON.parse(fs.readFileSync(graphPath, 'utf8'));
  const layers = JSON.parse(fs.readFileSync(layersPath, 'utf8'));
  
  const inputData = {
    graph,
    layers
  };
  
  // Ensure directory exists
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(inputData, null, 2));
  console.log('Successfully wrote input JSON to ' + outputPath);
} catch (error) {
  console.error('Error preparing input JSON:', error);
  process.exit(1);
}
