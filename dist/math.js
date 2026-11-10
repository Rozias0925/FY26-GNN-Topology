/* Pure arithmetic for the explicitly untrained, scalar teaching example. */
(function (root) {
  'use strict';
  function modelExample(model, time, features) {
    const dynamic = model.startsWith('Dynamic');
    const attention = model.endsWith('GAT');
    const neighbors = model === 'MLP' ? [] : dynamic && time === 1 ? [1, 3] : [1, 2];
    const active = [0, ...neighbors];
    let weights;
    if (model === 'MLP') weights = [1];
    else if (attention) {
      const scores = active.map(i => features[i] >= 0 ? features[i] : .2 * features[i]);
      const maximum = Math.max(...scores);
      const exponentials = scores.map(x => Math.exp(x - maximum));
      const denominator = exponentials.reduce((a, b) => a + b, 0);
      weights = exponentials.map(x => x / denominator);
    } else {
      const degree = active.length;
      weights = active.map((_, index) => index === 0 ? 1 / degree : 1 / Math.sqrt(2 * degree));
    }
    const terms = active.map((node, index) => ({ node, feature: features[node], weight: weights[index], contribution: weights[index] * features[node] }));
    const aggregate = terms.reduce((sum, term) => sum + term.contribution, 0);
    return { active, neighbors, terms, aggregate, output: Math.max(0, aggregate), attention, dynamic };
  }
  function conceptualEdges(demo, drift, time) {
    const count = time === 0 ? 0 : Math.round(drift / 100 * demo.alternatives.length);
    const removed = new Set(Array.from({ length: count }, (_, i) => (i * 2 + time) % demo.edges.length));
    return [...demo.edges.filter((_, i) => !removed.has(i)).map(edge => ({ edge, changed: false })),
      ...Array.from({ length: count }, (_, i) => ({ edge: demo.alternatives[(i + time) % demo.alternatives.length], changed: true }))];
  }
  root.GraphLesson = { modelExample, conceptualEdges };
  if (typeof module !== 'undefined') module.exports = root.GraphLesson;
})(typeof window !== 'undefined' ? window : globalThis);
