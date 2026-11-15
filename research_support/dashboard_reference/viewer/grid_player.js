/* Recorded-grid renderer; display only, no network or filesystem. */
(function(global){
  'use strict';
  function mount(output,payload){
    output.setAttribute("data-recorded-grid", "");
    let replayTimer=null;
    const txt=(tag,text,cls)=>{const e=document.createElement(tag);e.textContent=text;if(cls)e.className=cls;return e;};
    const value=v=>v==null?'—':String(v);
    function svgNode(tag,attrs,text){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs||{}))e.setAttribute(k,v);if(text!=null)e.textContent=text;return e;}
    function renderGrid(output,p){
  output.replaceChildren();output.append(txt('p',`${p.fixture?'SYNTHETIC FIXTURE. ':''}${p.identity_checks} Exporter verification: ${p.verification.status}.`));
  const grid=p.map,frameLabel=txt('p',''),controls=txt('div','','actions'),prev=txt('button','Previous'),play=txt('button','Play'),next=txt('button','Next'),slider=document.createElement('input');
  slider.type='range';slider.min=0;slider.max=p.frames.length-1;slider.value=0;slider.setAttribute('aria-label','Recorded frame');
  controls.append(prev,play,next,slider);const drawing=txt('div','','gridDrawing'),events=txt('pre','');output.append(controls,frameLabel,drawing,txt('p','P player · G goal · E enemy · H hazard · × obstacle · S special traversal. Tile numbers are recorded elevations. Unknown state is never synthesized.'),events);
  const setFrame=value=>{slider.value=Math.max(0,Math.min(p.frames.length-1,value));paint();};
  const paint=()=>{
    const frame=p.frames[Number(slider.value)],size=40;drawing.replaceChildren();
    const svg=svgNode('svg',{viewBox:`0 0 ${grid.width*size} ${grid.height*size}`,role:'img','aria-label':`Recorded map frame ${frame.step}`});
    grid.tiles.forEach((row,y)=>row.forEach((tile,x)=>{
      const g=svgNode('g',{}),color=tile.obstacle?'#6b7475':tile.hazard?'#f4c093':tile.special_traversal?'#c5d5e5':'#e7eddf';
      const r=svgNode('rect',{x:x*size,y:y*size,width:size,height:size,fill:color,stroke:'#889388'});r.append(svgNode('title',{},`(${x},${y}) ${JSON.stringify(tile)}`));
      g.append(r,svgNode('text',{x:x*size+3,y:y*size+12,'font-size':10},tile.elevation));
      const marks=[];if(tile.obstacle)marks.push('×');if(tile.hazard)marks.push('H');if(tile.special_traversal)marks.push('S');if(grid.goal[0]===x&&grid.goal[1]===y)marks.push('G');
      if(frame.enemy_position?.[0]===x&&frame.enemy_position?.[1]===y)marks.push('E');if(frame.agent_position[0]===x&&frame.agent_position[1]===y)marks.push('P');
      g.append(svgNode('text',{x:x*size+size/2,y:y*size+29,'text-anchor':'middle','font-size':14,fill:'#172e24'},marks.join('')));svg.append(g);
    }));drawing.append(svg);frameLabel.textContent=`Frame ${frame.step}/${p.frames.length-1} · HP ${frame.agent_hp} · action ${value(frame.action)} · raw reward ${value(frame.reward)} · end ${value(frame.end_reason)}`;
    events.textContent=JSON.stringify({events:frame.events,enemy_position:frame.enemy_position,enemy_patrol_index:frame.enemy_patrol_index,enemy_spec:p.enemy_spec,terminated:frame.terminated,truncated:frame.truncated},null,2);
  };
  slider.oninput=paint;prev.onclick=()=>setFrame(Number(slider.value)-1);next.onclick=()=>setFrame(Number(slider.value)+1);
  play.onclick=()=>{if(replayTimer){clearInterval(replayTimer);replayTimer=null;play.textContent='Play';return;}if(Number(slider.value)===p.frames.length-1)setFrame(0);play.textContent='Pause';replayTimer=setInterval(()=>{setFrame(Number(slider.value)+1);if(Number(slider.value)===p.frames.length-1){clearInterval(replayTimer);replayTimer=null;play.textContent='Play';}},500);};paint();
}

    renderGrid(output,payload);
    return {destroy(){clearInterval(replayTimer);replayTimer=null;output.replaceChildren();}};
  }
  global.GatorRecordedGrid={mount};
})(window);
