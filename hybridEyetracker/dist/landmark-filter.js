import {updateOneEuro2D} from "./tracker-math.js";

const EYES=new Set([33,133,159,145,160,144,158,153,161,163,157,154,362,263,386,374,385,380,387,373,388,390,384,381]);
export function stabilizeLandmarks(previous,raw,now,{width=640,height=480}={}){
  if(!Array.isArray(raw)||raw.length<468||!Number.isFinite(now)||!raw.every(p=>[p.x,p.y,p.z??0].every(Number.isFinite)))return null;
  const aspect=width/height;
  if(!Number.isFinite(aspect)||aspect<=0)return null;
  const xs=raw.slice(0,468).map(p=>p.x),ys=raw.slice(0,468).map(p=>p.y);
  const scale=Math.max(.05,Math.hypot((Math.max(...xs)-Math.min(...xs))*aspect,Math.max(...ys)-Math.min(...ys)));
  const center={x:(Math.max(...xs)+Math.min(...xs))/2,y:(Math.max(...ys)+Math.min(...ys))/2};
  const reset=!previous||now<=previous.at||now-previous.at>150||previous.width!==width||previous.height!==height||previous.filters.length!==raw.length||Math.hypot((center.x-previous.center.x)*aspect,center.y-previous.center.y)>scale*.5;
  const filters=[],points=raw.map((p,i)=>{
    const iris=i>=468,eye=EYES.has(i),config={minCutoff:iris?8:eye?4:1.5,beta:(iris?1.2:eye?.8:.5)/scale,maxGapMs:150};
    const old=reset?null:previous.filters[i];
    const xy=updateOneEuro2D(old?.xy,{x:p.x*aspect,y:p.y},now,config);
    const z=updateOneEuro2D(old?.z,{x:(p.z??0)*aspect,y:0},now,config);
    filters.push({xy:xy.state,z:z.state});
    return {...p,x:xy.point.x/aspect,y:xy.point.y,z:z.point.x/aspect};
  });
  return {points,state:{at:now,width,height,center,filters},reset};
}
