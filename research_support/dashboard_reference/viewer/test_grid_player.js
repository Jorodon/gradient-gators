'use strict';
// In-memory DOM interaction test. No browser, network or numerical execution.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const path=require('node:path'),crypto=require('node:crypto');
if(!process.env.DASHBOARD_RELEASES_ROOT){console.log('SKIP: set DASHBOARD_RELEASES_ROOT explicitly for the external synthetic fixture.');process.exit(0);}
const release=path.resolve(process.env.DASHBOARD_RELEASES_ROOT,'dashboard_synthetic_examples_v1');
const manifest=JSON.parse(fs.readFileSync(path.join(release,'release_manifest.json'),'utf8'));
const entry=manifest.files.find(e=>e.artifact_id==='synthetic_trajectory');
assert(entry,'Synthetic trajectory is not indexed');
const raw=fs.readFileSync(path.join(release,'payload','trajectory.json'));
assert.equal(raw.length,entry.bytes);assert.equal(crypto.createHash('sha256').update(raw).digest('hex'),entry.sha256);
const trace=JSON.parse(raw.toString('utf8'));
class Element {
  constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.attrs={};this.value='';}
  append(...items){this.children.push(...items);}
  replaceChildren(...items){this.children=items;}
  setAttribute(k,v){this.attrs[k]=v;}
}
const tree=e=>[e,...e.children.flatMap(tree)];
const text=e=>e.textContent+' '+e.children.map(text).join(' ');
const context={window:{},document:{createElement:t=>new Element(t),createElementNS:(_,t)=>new Element(t)},clearInterval,setInterval};
vm.createContext(context);
vm.runInContext(fs.readFileSync(__dirname+'/grid_player.js','utf8'),context);
// Geometry and frames come only from the explicitly configured external release.
const payload={fixture:true,map:trace.scenario_manifest.map,frames:trace.frames,enemy_spec:trace.scenario_manifest.enemy_spec,
  identity_checks:'synthetic fixture',verification:{status:'captured_original'}};
const first=new Element('div'),second=new Element('div');
const a=context.window.GatorRecordedGrid.mount(first,payload),b=context.window.GatorRecordedGrid.mount(second,payload);
assert.equal(tree(first).filter(e=>e.tag==='rect').length,9);
tree(first).find(e=>e.tag==='button'&&e.textContent==='Next').onclick();
assert(text(first).includes('Frame 1/2'));assert(text(first).includes('HP 10'));
assert(text(second).includes('Frame 0/2'),'instances must not share frame selection');
a.destroy();assert.equal(first.children.length,0);assert(second.children.length>0);b.destroy();
console.log('PASS: recorded grid, saved frame transition, independent mounts and cleanup. No browser visual QA claimed.');
