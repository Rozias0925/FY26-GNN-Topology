/* Rebuild the browser bundle from the existing verified JSON export.
 * This does not read, replace, or change permissions on mechanism-results-v2.js.
 */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const root = path.resolve(__dirname, '..');
const dist = path.join(root, 'dist');
const sourcePath = path.join(dist, 'data', 'mechanism-results-v2.json');
const outputPath = path.join(dist, 'data', 'mechanism-results-web.js');
const indexPath = path.join(dist, 'index.html');
const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

async function fileHash(filename) {
  const hash = crypto.createHash('sha256');
  for await (const chunk of fs.createReadStream(filename)) hash.update(chunk);
  return hash.digest('hex');
}

function artifactPath(relative) {
  assert.match(relative, /^data\/mechanisms-v2\/[A-Za-z0-9_./-]+\.(?:csv|json|png)$/);
  assert(!relative.split('/').includes('..'), 'Artifact path must remain within dist.');
  const absolute = path.resolve(dist, relative);
  assert(absolute.startsWith(dist + path.sep), 'Artifact escaped dist.');
  return absolute;
}

async function main() {
  const source = fs.readFileSync(sourcePath);
  const raw = source.toString('utf8');
  const data = JSON.parse(raw);
  assert.equal(data.schema_version, 2);
  assert.equal(data.status, 'complete');
  const names = ['main', 'convergence', 'robustness', 'stage_a'];
  assert.deepEqual(Object.keys(data.groups).sort(), [...names].sort());
  const counts = {};
  for (const name of names) {
    const group = data.groups[name];
    assert.equal(group.schema_version, 2);
    assert.equal(group.complete, true);
    assert(Number.isInteger(group.planned_conditions) && group.planned_conditions > 0);
    assert.equal(group.completed_conditions, group.planned_conditions);
    for (const key of ['model_results', 'contrasts', 'figures']) assert(Array.isArray(group[key]));
    const conditions = new Set(group.model_results.map(row => row.condition_id));
    assert.equal(conditions.size, group.completed_conditions);
    const evaluations = new Set(group.model_results.map(row => JSON.stringify([row.condition_id, row.seed, row.model])));
    counts[name] = {
      conditions: conditions.size,
      displayed_rows: group.model_results.length,
      unique_evaluations: evaluations.size,
      contrasts: group.contrasts.length,
      figures: group.figures.length
    };
  }
  const suiteCount = names.filter(name => name !== 'stage_a').reduce((sum, name) => sum + counts[name].unique_evaluations, 0);
  assert.equal(suiteCount, data.provenance.suite_model_evaluations);
  assert.equal(counts.stage_a.unique_evaluations, data.provenance.separate_stage_a_evaluations);

  const hashes = data.provenance.download_sha256;
  assert(hashes && Object.keys(hashes).length > 0);
  for (const [relative, expected] of Object.entries(hashes)) {
    assert.match(expected, /^[a-f0-9]{64}$/);
    assert.equal(await fileHash(artifactPath(relative)), expected, `SHA256 mismatch: ${relative}`);
  }
  assert(Array.isArray(data.downloads) && data.downloads.length > 0);
  const seenDownloads = new Set();
  for (const download of data.downloads) {
    assert(!seenDownloads.has(download.path), `Duplicate download: ${download.path}`);
    seenDownloads.add(download.path);
    const filename = artifactPath(download.path);
    if (download.path === 'data/mechanisms-v2/provenance.json') {
      // A provenance file cannot include its own content hash. Compare its JSON instead.
      assert.deepEqual(JSON.parse(fs.readFileSync(filename, 'utf8')), data.provenance);
    } else assert(Object.hasOwn(hashes, download.path), `Unverified download: ${download.path}`);
  }
  for (const group of Object.values(data.groups)) {
    for (const figure of group.figures) assert(Object.hasOwn(hashes, figure), `Unverified figure: ${figure}`);
  }

  // Wrap the unchanged JSON text; numerical literals are never reserialized.
  // Escape HTML and line separators so the result is safe as a static script.
  const quoted = JSON.stringify(raw).replace(/</g, '\\u003c').replace(/\u2028/g, '\\u2028').replace(/\u2029/g, '\\u2029');
  const script = `/* Source JSON SHA256: ${sha256(source)} */\nwindow.MECHANISM_RESULTS = JSON.parse(${quoted});\n`;
  const browserWindow = {};
  vm.runInNewContext(script, {window: browserWindow, JSON}, {timeout: 10000});
  assert.deepEqual(browserWindow.MECHANISM_RESULTS, data, 'Browser data differs from the JSON export.');

  fs.writeFileSync(outputPath, script, 'utf8');
  const index = fs.readFileSync(indexPath, 'utf8');
  const scriptTag = /src="data\/mechanism-results-(?:v2|web)\.js(?:\?v=\d+)?"/g;
  assert.equal([...index.matchAll(scriptTag)].length, 1, 'Expected exactly one results script tag.');
  const updated = index.replace(scriptTag, 'src="data/mechanism-results-web.js?v=1"');
  if (updated !== index) fs.writeFileSync(indexPath, updated, 'utf8');
  console.log(JSON.stringify({
    source_sha256: sha256(source),
    browser_script_sha256: sha256(Buffer.from(script, 'utf8')),
    schema_version: data.schema_version,
    status: data.status,
    suite_evaluations: suiteCount,
    separate_stage_a_evaluations: counts.stage_a.unique_evaluations,
    groups: counts,
    verified_sha256_files: Object.keys(hashes).length,
    verified_downloads: data.downloads.length,
    browser_data_deep_equal: true,
    output: path.relative(root, outputPath).replaceAll(path.sep, '/')
  }));
}

main().catch(error => {console.error(error.message); process.exitCode = 1;});
