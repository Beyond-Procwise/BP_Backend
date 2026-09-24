// Exports the UI's current copy to flat JSON, READ-ONLY against the UI repo.
// A bridge until the UI owns locales/en.json itself (see the hand-off spec).
//   node --experimental-specifier-resolution=node scripts/i18n_export_ui_catalog.mjs \
//        --ui /home/muthu/PycharmProjects/beyond_procwise_ui --out <dir>
// Writes <out>/en.json and <out>/{es,fr,de}.json. Keys: the UI's named keys as-is, and
// SpendIQ body phrases as "tb:" + the English sentence (they are keyed by their English).
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const arg = (name) => { const i = process.argv.indexOf(name); return i > 0 ? process.argv[i + 1] : undefined; };
const ui = arg('--ui'); const out = arg('--out');
if (!ui || !out) { console.error('usage: --ui <ui repo> --out <dir>'); process.exit(2); }

const { STRINGS } = await import(join(ui, 'src/lib/i18n/strings.js'));
const { BODY } = await import(join(ui, 'src/lib/i18n/body.js'));

const en = { ...STRINGS.en };
const langs = Object.keys(BODY);
for (const phrase of Object.keys(BODY[langs[0]] || {})) en[`tb:${phrase}`] = phrase;

mkdirSync(out, { recursive: true });
writeFileSync(join(out, 'en.json'), JSON.stringify(en, null, 1));
for (const lang of langs) {
  const tr = {};
  for (const [k, v] of Object.entries(STRINGS[lang] || {})) if (k in STRINGS.en) tr[k] = v;
  for (const [phrase, v] of Object.entries(BODY[lang] || {})) tr[`tb:${phrase}`] = v;
  writeFileSync(join(out, `${lang}.json`), JSON.stringify(tr, null, 1));
  console.log(lang, Object.keys(tr).length, 'of', Object.keys(en).length);
}
