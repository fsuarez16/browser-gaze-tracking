export function fitEllipse(points) {
  points=(points||[]).filter(p=>p&&Number.isFinite(p.x)&&Number.isFinite(p.y));
  if (points.length<3) return{center:{x:NaN,y:NaN},major:0,minor:0,angle:0};
  const cx=points.reduce((s,p)=>s+p.x,0)/points.length,cy=points.reduce((s,p)=>s+p.y,0)/points.length;
  let xx=0,xy=0,yy=0;for(const p of points){const x=p.x-cx,y=p.y-cy;xx+=x*x;xy+=x*y;yy+=y*y}xx/=points.length;xy/=points.length;yy/=points.length;
  const angle=.5*Math.atan2(2*xy,xx-yy),trace=xx+yy,root=Math.sqrt(Math.max(0,(xx-yy)**2+4*xy*xy));
  return{center:{x:cx,y:cy},major:2*Math.sqrt(Math.max(0,(trace+root)/2)),minor:2*Math.sqrt(Math.max(0,(trace-root)/2)),angle};
}
export function headPose(matrix,interocularNormalized,frameWidth,fovDegrees=43) {
  const m=matrix?.data||[],focalPx=frameWidth/(2*Math.tan(fovDegrees*Math.PI/360)),distanceMm=60*focalPx/Math.max(1,interocularNormalized*frameWidth);
  return{rotation:m.length>=16?[m[0],m[1],m[2],m[4],m[5],m[6],m[8],m[9],m[10]]:[1,0,0,0,1,0,0,0,1],translation:m.length>=16?[m[3]/100,m[7]/100,m[11]/100]:[0,0,0],distanceMm,focalPx};
}
export function eyeRayFromOffsets(offsetX,offsetY,pose,gain=1,side=-1) {
  const radiusMm=12,dx=offsetX*gain,dy=offsetY*gain;
  const radial=Math.min(.82,Math.hypot(dx,dy)),surfaceZ=Math.sqrt(Math.max(0,1-radial*radial));
  let x=dx,y=dy,z=surfaceZ,n=Math.hypot(x,y,z);x/=n;y/=n;z/=n;
  const r=pose.rotation,world={x:r[0]*x+r[1]*y+r[2]*z,y:r[3]*x+r[4]*y+r[5]*z,z:r[6]*x+r[7]*y+r[8]*z};
  const halfInterocularMm=30,offset={x:r[0]*side*halfInterocularMm,y:r[3]*side*halfInterocularMm,z:r[6]*side*halfInterocularMm},eyeCenter={x:pose.translation[0]+offset.x,y:pose.translation[1]+offset.y,z:pose.distanceMm+pose.translation[2]+offset.z},pupil={x:eyeCenter.x+world.x*radiusMm,y:eyeCenter.y+world.y*radiusMm,z:eyeCenter.z-radiusMm*world.z};
  const t=-pupil.z/(Math.abs(world.z)<.05?(world.z<0?-.05:.05):world.z);return{radiusMm,eyeCenter,pupil,direction:world,screen:{x:pupil.x+world.x*t,y:pupil.y+world.y*t,z:pupil.z+world.z*t}};
}
export function eyeRay(ellipse,cornerA,cornerB,pose,gain=1,side=-1) {
  const width=Math.hypot(cornerB.x-cornerA.x,cornerB.y-cornerA.y)||1,mx=(cornerA.x+cornerB.x)/2,my=(cornerA.y+cornerB.y)/2;
  return eyeRayFromOffsets((ellipse.center.x-mx)/width,(ellipse.center.y-my)/width,pose,gain,side);
}
export function optimizeFocalGain(samples,score) {
  let best={fov:43,gain:1,error:Infinity};const evaluate=score||((fov,gain)=>samples.reduce((sum,s)=>{const scale=gain*(43/fov),x=.5+scale*((s.f[0]+s.f[1])/2-.5),y=.5+scale*((s.f[2]+s.f[3])/2);return sum+Math.hypot(x-s.x,y-s.y)},0)/samples.length);
  for(let i=0;i<30;i++){const fov=70-i*60/29;for(let g=0;g<=10;g++){const gain=.5+g*.1,error=evaluate(fov,gain);if(error<best.error)best={fov,gain,error}}}
  for(const step of[1,.2])for(let fi=-5;fi<=5;fi++)for(let gi=-5;gi<=5;gi++){const fov=Math.max(10,Math.min(70,best.fov+fi*step)),gain=Math.max(.5,Math.min(1.5,best.gain+gi*step/5)),error=evaluate(fov,gain);if(error<best.error)best={fov,gain,error}}
  return best;
}
export function recoveryWindowStatus(points,minFrames=5,maxDispersion=.08){
  const valid=(points||[]).filter(point=>point&&[point.x,point.y].every(Number.isFinite));
  if(valid.length<minFrames)return{ready:false,stable:false,point:null,dispersion:Infinity};
  const median=value=>{const sorted=[...value].sort((a,b)=>a-b);return sorted[Math.floor(sorted.length/2)]},point={x:median(valid.map(row=>row.x)),y:median(valid.map(row=>row.y))},dispersion=Math.max(...valid.map(row=>Math.hypot(row.x-point.x,row.y-point.y)));
  return{ready:true,stable:Number.isFinite(dispersion)&&dispersion<=maxDispersion,point,dispersion};
}
export function faceTrackingQuality(width,eyeVisibility,centerDistance){
  if(![width,eyeVisibility,centerDistance].every(Number.isFinite))return 0;
  const clamp=value=>Math.max(0,Math.min(1,value)),size=clamp((width-.08)/.16),center=1-clamp(centerDistance/.55),eyes=clamp(eyeVisibility/.025);
  return clamp(size*(.55+.30*center+.15*eyes));
}
export class OneEuroFilter {
  constructor(minCutoff=1,beta=.035,dCutoff=1){this.minCutoff=minCutoff;this.beta=beta;this.dCutoff=dCutoff;this.last=null;this.derivative=0;this.time=null}
  reset(){this.last=null;this.derivative=0;this.time=null}
  alpha(cutoff,dt){const tau=1/(2*Math.PI*cutoff);return 1/(1+tau/dt)}
  filter(value,time=performance.now()/1000){if(!Number.isFinite(value))return Number.isFinite(this.last)?this.last:NaN;if(!Number.isFinite(time))time=performance.now()/1000;if(!Number.isFinite(this.last)||!Number.isFinite(this.derivative)||!Number.isFinite(this.time))this.reset();if(this.last===null){this.last=value;this.time=time;return value}const dt=Math.max(1/120,time-this.time),rawDerivative=(value-this.last)/dt,da=this.alpha(this.dCutoff,dt);this.derivative+=da*(rawDerivative-this.derivative);const cutoff=this.minCutoff+this.beta*Math.abs(this.derivative),a=this.alpha(cutoff,dt);this.last+=a*(value-this.last);this.time=time;return this.last}
}
