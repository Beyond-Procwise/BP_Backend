// Hands the UI's English catalogue to the pre-translation scripts, READ-ONLY against the UI repo.
//   node scripts/i18n_export_ui_catalog.mjs --ui /path/to/beyond_procwise_ui --out <dir>
// Writes <out>/en.json. English only, by design -- see below.
//
// This used to rebuild the catalogue itself, out of the UI's strings.js and body.js, and it
// got a narrower answer than the product actually uses: 1,921 keys against the 2,668 the UI
// asks the service for. It never saw a tOr() call site, a module tile or a body phrase too
// long to be its own key. So roughly 750 keys -- the whole Action Centre among them -- were
// missing from every pre-translation run, in every language, and only ever got translated
// when a user happened to open the screen and wait.
//
// The UI already builds the real thing at src/locales/en.json (scripts/generateEnJson.mjs,
// with a test that fails when it drifts). There is no second catalogue worth deriving, so
// this reads that file rather than guessing at it.
//
// It also no longer writes es/fr/de. Translations belong to the service and its store, not
// to a dictionary shipped in the UI bundle; the human-reviewed copy that used to live there
// is in proc.bp_translation as origin='reviewed', and scripts/i18n_import_reviewed.py is how
// more of it gets there.
import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { isAbsolute, join, resolve } from 'node:path';

const arg = (name) => { const i = process.argv.indexOf(name); return i > 0 ? process.argv[i + 1] : undefined; };
const ui = arg('--ui'); const out = arg('--out');
if (!ui || !out) { console.error('usage: --ui <ui repo> --out <dir>'); process.exit(2); }

/* Resolved against the shell's cwd, not this file's. A relative --ui used to be resolved as
   an import specifier relative to scripts/, so `--ui ../beyond_procwise_ui` looked inside
   BP_Backend and died with ERR_MODULE_NOT_FOUND. */
const at = (p) => (isAbsolute(p) ? p : resolve(process.cwd(), p));
const source = join(at(ui), 'src/locales/en.json');

let en;
try {
  en = JSON.parse(readFileSync(source, 'utf8'));
} catch (err) {
  console.error(`cannot read the UI catalogue at ${source}: ${err.message}`);
  console.error('build it first: npx vite-node scripts/generateEnJson.mjs (in the UI repo)');
  process.exit(2);
}

mkdirSync(at(out), { recursive: true });
writeFileSync(join(at(out), 'en.json'), JSON.stringify(en, null, 1));
console.log(`en ${Object.keys(en).length} keys -> ${join(at(out), 'en.json')}`);
