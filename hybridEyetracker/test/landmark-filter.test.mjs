import test from "node:test";
import assert from "node:assert/strict";
import {stabilizeLandmarks} from "../dist/landmark-filter.js";
const face=()=>Array.from({length:478},(_,i)=>({x:.5+.18*Math.cos(i),y:.5+.25*Math.sin(i),z:.02*Math.cos(i)}));
test("landmark filter reduces jitter and preserves more iris movement",()=>{
  let state=null;const base=face();let rawEnergy=0,filteredEnergy=0;
  for(let i=0;i<90;i++){const raw=base.map(p=>({...p,x:p.x+(i%2?.003:-.003)}));const out=stabilizeLandmarks(state,raw,i*33);state=out.state;if(i>15){rawEnergy+=(raw[10].x-base[10].x)**2;filteredEnergy+=(out.points[10].x-base[10].x)**2}}
  assert.ok(filteredEnergy<rawEnergy*.4);
  const moved=base.map(p=>({...p,x:p.x+.015})),out=stabilizeLandmarks(state,moved,90*33);
  assert.ok(out.points[468].x-base[468].x>out.points[10].x-base[10].x);
  assert.equal(base[10].x,face()[10].x);
});
test("landmark filter resets on gaps camera changes and rejects malformed data",()=>{
 const raw=face(),first=stabilizeLandmarks(null,raw,0);
 for(const [now,size] of [[200,{}],[33,{width:1280,height:720}]]){const out=stabilizeLandmarks(first.state,raw,now,size);assert.equal(out.reset,true);out.points.forEach((p,i)=>{assert.ok(Math.abs(p.x-raw[i].x)<1e-12);assert.ok(Math.abs(p.y-raw[i].y)<1e-12);assert.ok(Math.abs(p.z-raw[i].z)<1e-12)})}
 raw[0].x=NaN;assert.equal(stabilizeLandmarks(null,raw,20),null);
});
test("isolated iris motion remains responsive while the face stays stationary",()=>{
 const raw=face();let out=stabilizeLandmarks(null,raw,0);
 for(let i=1;i<=15;i++)out=stabilizeLandmarks(out.state,raw,i*33);
 const moved=raw.map(p=>({...p}));moved[468].x+=.02;
 const next=stabilizeLandmarks(out.state,moved,16*33);
 assert.ok(next.points[468].x-raw[468].x>.01);
 assert.ok(Math.abs(next.points[10].x-raw[10].x)<1e-12);
 assert.equal(stabilizeLandmarks(next.state,moved,10).reset,true);
});
