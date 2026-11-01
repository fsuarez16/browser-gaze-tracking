export function pixelError(dxNormalized,dyNormalized,width,height){
  return Math.hypot(dxNormalized*width,dyNormalized*height);
}

export function diagonalError(dxNormalized,dyNormalized,width,height){
  return pixelError(dxNormalized,dyNormalized,width,height)/Math.hypot(width,height);
}

export function createAdaptiveDriftModel(){const n=6;return{schema:1,thetaX:Array(n).fill(0),thetaY:Array(n).fill(0),P:Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?12:0)),count:0,accepted:0,rejected:0,revision:0,mode:"offset",anchors:[]}}
export function adaptiveDriftFeatures(base,head={}){return[1,(base?.x??.5)-.5,(base?.y??.5)-.5,Math.max(-1.5,Math.min(1.5,(head.yaw||0)/26)),Math.max(-1.5,Math.min(1.5,(head.pitch||0)/22)),Math.max(-1.5,Math.min(1.5,(head.z||0)/3))]}
export function validateAdaptiveDriftModel(model){return Boolean(model?.schema===1&&model.count>=0&&model.count<=10000&&[model.thetaX,model.thetaY].every(v=>Array.isArray(v)&&v.length===6&&v.every(Number.isFinite))&&Array.isArray(model.P)&&model.P.length===6&&model.P.every(row=>Array.isArray(row)&&row.length===6&&row.every(Number.isFinite)))}
export function predictAdaptiveDrift(model,base,head={}){if(!validateAdaptiveDriftModel(model))return{x:0,y:0,features:adaptiveDriftFeatures(base,head)};const features=adaptiveDriftFeatures(base,head),x=model.thetaX.reduce((s,v,i)=>s+v*features[i],0),y=model.thetaY.reduce((s,v,i)=>s+v*features[i],0),length=Math.hypot(x,y),scale=length>.12?.12/length:1;return{x:x*scale,y:y*scale,features}}
export function updateAdaptiveDriftRls(previous,anchor,{lambda=.995,huber=.04,maxStep=.02,maxResidual=.15}={}){const model=validateAdaptiveDriftModel(previous)?structuredClone(previous):createAdaptiveDriftModel(),base=anchor?.base,target=anchor?.target,head=anchor?.head||{};if(!base||!target||![base.x,base.y,target.x,target.y].every(Number.isFinite))return{model,accepted:false,reason:"invalid-anchor"};const desired={x:target.x-base.x,y:target.y-base.y},magnitude=Math.hypot(desired.x,desired.y);if(magnitude>maxResidual)return{model:{...model,rejected:model.rejected+1},accepted:false,reason:"residual-too-large"};const full=adaptiveDriftFeatures(base,head),evidence=[...(model.anchors||[]),{base,head}],span=values=>Math.max(...values)-Math.min(...values),xs=evidence.map(a=>a.base.x),ys=evidence.map(a=>a.base.y),yaws=evidence.map(a=>(a.head?.yaw||0)/26),pitches=evidence.map(a=>(a.head?.pitch||0)/22),zs=evidence.map(a=>(a.head?.z||0)/3),spatialReady=evidence.length>=4&&span(xs)>.12&&span(ys)>.12,poseReady=evidence.length>=7&&span(yaws)>.18&&span(pitches)>.18,zReady=evidence.length>=7&&span(zs)>.12,spatialActive=spatialReady&&["spatial-ready","spatial-rls","pose-ready","pose-rls"].includes(model.mode),poseActive=(poseReady||zReady)&&["pose-ready","pose-rls"].includes(model.mode),mode=poseActive?"pose-rls":poseReady||zReady?"pose-ready":spatialActive?"spatial-rls":spatialReady?"spatial-ready":"offset",phi=full.map((v,i)=>i===0||(spatialActive&&(i===1||i===2))||(poseActive&&poseReady&&(i===3||i===4))||(poseActive&&zReady&&i===5)?v:0),Pphi=model.P.map(row=>row.reduce((s,v,i)=>s+v*phi[i],0)),den=lambda+phi.reduce((s,v,i)=>s+v*Pphi[i],0),K=Pphi.map(v=>v/den),before=predictAdaptiveDrift(model,base,head),rx=Math.max(-huber,Math.min(huber,desired.x-before.x)),ry=Math.max(-huber,Math.min(huber,desired.y-before.y));let dx=K.map(v=>v*rx),dy=K.map(v=>v*ry),stepX=dx.reduce((s,v,i)=>s+v*phi[i],0),stepY=dy.reduce((s,v,i)=>s+v*phi[i],0),step=Math.hypot(stepX,stepY),scale=step>maxStep?maxStep/step:1,effectiveK=K.map(v=>v*scale);dx=dx.map(v=>v*scale);dy=dy.map(v=>v*scale);const thetaX=model.thetaX.map((v,i)=>v+dx[i]),thetaY=model.thetaY.map((v,i)=>v+dy[i]),P=model.P.map((row,i)=>row.map((v,j)=>(v-effectiveK[i]*Pphi[j])/lambda)),next={...model,thetaX,thetaY,P,count:model.count+1,accepted:model.accepted+1,revision:model.revision+1,mode,anchors:[...(model.anchors||[]),{base:{...base},target:{...target},head:{...head},confidence:anchor.confidence,at:anchor.at}].slice(-40)};return{model:next,accepted:true,before,after:predictAdaptiveDrift(next,base,head),desired,step:Math.hypot(stepX*scale,stepY*scale)}}

function fixationEvent(f,now,width,height,minFixationMs){
  if(!f||f.count<3||now-f.started<minFixationMs)return null;
  return{type:"fixation",x:f.x/width,y:f.y/height,startedAt:f.started,endedAt:now,dwellMs:Math.round(now-f.started),samples:f.count,meanConfidence:f.confidenceSum/f.count,dispersionPx:(f.maxX-f.minX)+(f.maxY-f.minY),dispersionMetric:"range-sum"};
}

export function updateGazeEventClassifier(previous,sample,{width,height,minFixationMs=120,dispersionRatio=.025,highVelocity=0.65,lowVelocity=0.36,settleSamples=2}={}){
  if(!sample||![sample.x,sample.y,sample.at,sample.confidence,width,height].every(Number.isFinite)||width<=0||height<=0)return{state:previous,events:[]};
  const diag=Math.hypot(width,height),radius=Math.max(18,Math.min(width,height)*dispersionRatio),point={x:sample.x,y:sample.y,at:sample.at,confidence:sample.confidence};
  if(sample.confidence<.25)return{state:previous,events:[],rejectedLowConfidence:1};
  if(!previous?.last){return{state:{mode:"fixation",last:point,fixation:{x:sample.x,y:sample.y,started:sample.at,count:1,confidenceSum:sample.confidence,minX:sample.x,maxX:sample.x,minY:sample.y,maxY:sample.y,lastPaint:0},saccade:null,settled:[],rejectedLowConfidence:0},events:[]}}
  if(sample.at-previous.last.at>250){const flushed=flushGazeEventClassifier(previous,previous.last.at,{width,height,minFixationMs});return{state:{mode:"fixation",last:point,fixation:{x:sample.x,y:sample.y,started:sample.at,count:1,confidenceSum:sample.confidence,minX:sample.x,maxX:sample.x,minY:sample.y,maxY:sample.y,lastPaint:0},saccade:null,settled:[],rejectedLowConfidence:previous.rejectedLowConfidence||0},events:flushed.events,discontinuity:true}}
  const dt=Math.max(.001,(sample.at-previous.last.at)/1000),distance=Math.hypot(sample.x-previous.last.x,sample.y-previous.last.y),velocity=distance/diag/dt,quality=Math.max(0,Math.min(1,sample.confidence)),startThreshold=highVelocity+(1-quality)*.25,endThreshold=lowVelocity+(1-quality)*.12,events=[],state={...previous,last:point};
  if(previous.mode==="fixation"){
    const f=previous.fixation,dispersion=(Math.max(f.maxX,sample.x)-Math.min(f.minX,sample.x))+(Math.max(f.maxY,sample.y)-Math.min(f.minY,sample.y));
    if(velocity>startThreshold||dispersion>radius*2){const event=fixationEvent(f,previous.last.at,width,height,minFixationMs);if(event)events.push(event);state.mode="saccade";state.fixation=null;state.settled=[];state.saccade={started:previous.last.at,startX:previous.last.x,startY:previous.last.y,endX:sample.x,endY:sample.y,peakVelocity:velocity,confidenceSum:quality,count:1,dispersionThresholdPx:radius*2};return{state,events}}
    const count=f.count+1;state.fixation={...f,count,x:f.x+(sample.x-f.x)/count,y:f.y+(sample.y-f.y)/count,confidenceSum:f.confidenceSum+quality,minX:Math.min(f.minX,sample.x),maxX:Math.max(f.maxX,sample.x),minY:Math.min(f.minY,sample.y),maxY:Math.max(f.maxY,sample.y)};return{state,events}
  }
  const s=previous.saccade||{started:previous.last.at,startX:previous.last.x,startY:previous.last.y,peakVelocity:0,confidenceSum:0,count:0},candidateSettled=velocity<endThreshold&&quality>.35?[...(previous.settled||[]),point]:[],settleDispersion=candidateSettled.length?(Math.max(...candidateSettled.map(p=>p.x))-Math.min(...candidateSettled.map(p=>p.x)))+(Math.max(...candidateSettled.map(p=>p.y))-Math.min(...candidateSettled.map(p=>p.y))):0,settled=settleDispersion<=(s.dispersionThresholdPx||radius*2)?candidateSettled:[point];state.saccade={...s,endX:sample.x,endY:sample.y,peakVelocity:Math.max(s.peakVelocity,velocity),confidenceSum:s.confidenceSum+quality,count:s.count+1};state.settled=settled;
  if(settled.length>=settleSamples){const endpoint=settled[0],amplitudePx=Math.hypot(endpoint.x-s.startX,endpoint.y-s.startY),durationMs=Math.max(1,Math.round(endpoint.at-s.started));events.push({type:"saccade",startX:s.startX/width,startY:s.startY/height,endX:endpoint.x/width,endY:endpoint.y/height,startedAt:s.started,endedAt:endpoint.at,durationMs,amplitudePx,amplitudeNormalized:amplitudePx/diag,directionDeg:Math.atan2(endpoint.y-s.startY,endpoint.x-s.startX)*180/Math.PI,peakVelocityNormalizedPerSec:state.saccade.peakVelocity,meanConfidence:state.saccade.confidenceSum/Math.max(1,state.saccade.count)});const count=settled.length,stateX=settled.reduce((sum,p)=>sum+p.x,0)/count,stateY=settled.reduce((sum,p)=>sum+p.y,0)/count;state.mode="fixation";state.last=point;state.fixation={x:stateX,y:stateY,started:endpoint.at,count,confidenceSum:settled.reduce((sum,p)=>sum+p.confidence,0),minX:Math.min(...settled.map(p=>p.x)),maxX:Math.max(...settled.map(p=>p.x)),minY:Math.min(...settled.map(p=>p.y)),maxY:Math.max(...settled.map(p=>p.y)),lastPaint:0};state.saccade=null;state.settled=[]}
  return{state,events};
}

export function flushGazeEventClassifier(previous,now,{width,height,minFixationMs=120}={}){
  if(!previous)return{state:null,events:[]};const events=[];
  if(previous.mode==="fixation"){const event=fixationEvent(previous.fixation,previous.last.at,width,height,minFixationMs);if(event)events.push({...event,flushedAt:now})}
  else if(previous.saccade){const s=previous.saccade,endedAt=previous.last.at,amplitudePx=Math.hypot((s.endX??previous.last.x)-s.startX,(s.endY??previous.last.y)-s.startY),diag=Math.hypot(width,height);events.push({type:"saccade",startX:s.startX/width,startY:s.startY/height,endX:(s.endX??previous.last.x)/width,endY:(s.endY??previous.last.y)/height,startedAt:s.started,endedAt,durationMs:Math.max(1,Math.round(endedAt-s.started)),flushedAt:now,amplitudePx,amplitudeNormalized:amplitudePx/diag,directionDeg:Math.atan2((s.endY??previous.last.y)-s.startY,(s.endX??previous.last.x)-s.startX)*180/Math.PI,peakVelocityNormalizedPerSec:s.peakVelocity,meanConfidence:s.confidenceSum/Math.max(1,s.count),incomplete:true})}
  return{state:null,events};
}

export function gazeBubbleGate({enabled=true,tracking=false,hasPoint=false,lastValidAt=0,now=0,reason=""}={}, {holdMs=260}={}){
  if(!enabled||!tracking||!hasPoint)return{visible:false,holding:false,age:Infinity};
  const age=Math.max(0,now-lastValidAt),terminal=reason==="not-calibrated";
  if(!reason)return{visible:true,holding:false,age};
  if(!terminal&&age<=holdMs)return{visible:true,holding:true,age};
  return{visible:false,holding:false,age};
}

export function updateOneEuro2D(previous,point,now,{minCutoff=1.2,beta=.035,derivativeCutoff=1,maxGapMs=150,confidence=1}={}){
  if(!point||![point.x,point.y,now,minCutoff,beta,derivativeCutoff,maxGapMs,confidence].every(Number.isFinite)||minCutoff<=0||derivativeCutoff<=0||maxGapMs<=0)return null;
  const reset=!previous||!Number.isFinite(previous.at)||now<=previous.at||now-previous.at>maxGapMs;
  if(reset)return{point:{x:point.x,y:point.y},state:{at:now,raw:{x:point.x,y:point.y},filtered:{x:point.x,y:point.y},derivative:{x:0,y:0}}};
  const dt=Math.max(.001,(now-previous.at)/1000),alpha=cutoff=>{const tau=1/(2*Math.PI*cutoff);return 1/(1+tau/dt)},dAlpha=alpha(derivativeCutoff),quality=Math.max(0,Math.min(1,confidence)),axis=key=>{const velocity=(point[key]-previous.raw[key])/dt,derivative=previous.derivative[key]+dAlpha*(velocity-previous.derivative[key]),cutoff=minCutoff*(.55+.45*quality)+beta*quality*Math.abs(derivative),filtered=previous.filtered[key]+alpha(cutoff)*(point[key]-previous.filtered[key]);return{derivative,filtered}};
  const x=axis("x"),y=axis("y"),state={at:now,raw:{x:point.x,y:point.y},filtered:{x:x.filtered,y:y.filtered},derivative:{x:x.derivative,y:y.derivative}};return{point:{...state.filtered},state};
}

export const PYTHON_NOSE_INDICES=[4,45,275,220,440,1,5,51,281,44,274,241,461,125,354,218,438,195,167,393,165,391,3,248];
const v3=(a,b)=>({x:a.x-b.x,y:a.y-b.y,z:a.z-b.z}),vadd=(a,b)=>({x:a.x+b.x,y:a.y+b.y,z:a.z+b.z}),vmul=(a,s)=>({x:a.x*s,y:a.y*s,z:a.z*s}),vdot=(a,b)=>a.x*b.x+a.y*b.y+a.z*b.z,vcross=(a,b)=>({x:a.y*b.z-a.z*b.y,y:a.z*b.x-a.x*b.z,z:a.x*b.y-a.y*b.x}),vlen=a=>Math.hypot(a.x,a.y,a.z),vnorm=a=>{const n=vlen(a);return n>1e-9?vmul(a,1/n):null};
export function meanPairwiseScale(points){if(!Array.isArray(points)||points.length<2)return null;let sum=0,n=0;for(let i=0;i<points.length;i++)for(let j=i+1;j<points.length;j++){const d=vlen(v3(points[i],points[j]));if(Number.isFinite(d)){sum+=d;n++}}return n?sum/n:null}
function matVec(m,v){return{x:m[0][0]*v.x+m[0][1]*v.y+m[0][2]*v.z,y:m[1][0]*v.x+m[1][1]*v.y+m[1][2]*v.z,z:m[2][0]*v.x+m[2][1]*v.y+m[2][2]*v.z}}
function transpose(m){return[[m[0][0],m[1][0],m[2][0]],[m[0][1],m[1][1],m[2][1]],[m[0][2],m[1][2],m[2][2]]]}
export function symmetricEigen3(matrix,{iterations=18}={}){if(!Array.isArray(matrix)||matrix.length!==3||matrix.some(r=>!Array.isArray(r)||r.length!==3||r.some(x=>!Number.isFinite(x))))return null;const a=matrix.map(r=>r.slice()),q=[[1,0,0],[0,1,0],[0,0,1]];for(let k=0;k<iterations;k++){let p=0,r=1;if(Math.abs(a[0][2])>Math.abs(a[p][r]))[p,r]=[0,2];if(Math.abs(a[1][2])>Math.abs(a[p][r]))[p,r]=[1,2];if(Math.abs(a[p][r])<1e-12)break;const phi=.5*Math.atan2(2*a[p][r],a[r][r]-a[p][p]),c=Math.cos(phi),s=Math.sin(phi);for(let i=0;i<3;i++){const aip=a[i][p],air=a[i][r];a[i][p]=c*aip-s*air;a[i][r]=s*aip+c*air}for(let i=0;i<3;i++){const api=a[p][i],ari=a[r][i];a[p][i]=c*api-s*ari;a[r][i]=s*api+c*ari;const qip=q[i][p],qir=q[i][r];q[i][p]=c*qip-s*qir;q[i][r]=s*qip+c*qir}}const order=[0,1,2].sort((i,j)=>a[j][j]-a[i][i]);return{values:order.map(i=>a[i][i]),vectors:order.map(i=>({x:q[0][i],y:q[1][i],z:q[2][i]}))}}
export function nosePcaFrame(landmarks,width,height,reference=null,{minEigenGap=.015}={}){if(!Array.isArray(landmarks)||!width||!height)return null;const points=PYTHON_NOSE_INDICES.map(i=>landmarks[i]).filter(Boolean).map(p=>({x:p.x*width,y:p.y*height,z:p.z*width}));if(points.length!==PYTHON_NOSE_INDICES.length)return null;const center=vmul(points.reduce(vadd,{x:0,y:0,z:0}),1/points.length),cov=[[0,0,0],[0,0,0],[0,0,0]];for(const p of points){const d=v3(p,center),v=[d.x,d.y,d.z];for(let i=0;i<3;i++)for(let j=0;j<3;j++)cov[i][j]+=v[i]*v[j]/Math.max(1,points.length-1)}const eig=symmetricEigen3(cov);if(!eig||eig.values[1]<1e-8)return null;const spread=Math.max(1e-8,Math.abs(eig.values[0])),gap01=(eig.values[0]-eig.values[1])/spread,gap12=(eig.values[1]-eig.values[2])/spread;if(Math.min(gap01,gap12)<minEigenGap)return null;let axes=eig.vectors.map(vnorm);if(reference?.axes)axes=axes.map((a,i)=>vdot(a,reference.axes[i])<0?vmul(a,-1):a);if(vdot(vcross(axes[0],axes[1]),axes[2])<0)axes[2]=vmul(axes[2],-1);const matrix=[[axes[0].x,axes[1].x,axes[2].x],[axes[0].y,axes[1].y,axes[2].y],[axes[0].z,axes[1].z,axes[2].z]],scale=meanPairwiseScale(points);return center&&scale>1e-6?{center,axes,matrix,scale,points,eigenvalues:eig.values,eigenGaps:[gap01,gap12]}:null}
export function lockEyeSpheres(landmarks,width,height,frame,{radius=20}={}){if(!frame||!landmarks?.[468]||!landmarks?.[473]||frame.scale<=1e-6)return null;const local=transpose(frame.matrix),cameraLocal=matVec(local,{x:0,y:0,z:1}),make=i=>vadd(matVec(local,v3({x:landmarks[i].x*width,y:landmarks[i].y*height,z:landmarks[i].z*width},frame.center)),vmul(cameraLocal,radius));return{leftOffset:make(468),rightOffset:make(473),calibrationScale:frame.scale,calibrationCenter:{...frame.center},calibrationAxes:frame.axes.map(a=>({...a})),radius,createdAt:Date.now(),source:"MonitorTracking.py eye-sphere port"}}
export function eyeSphereLockInvalidationReason(lock,frame,{faceLostMs=0,maxFaceLostMs=900,maxScaleChange=.32,maxCenterShift=1.25,minAxisDot=.82}={}){if(!lock)return null;if(faceLostMs>maxFaceLostMs)return"sustained-face-loss";if(!frame)return null;const ratio=frame.scale/lock.calibrationScale;if(!Number.isFinite(ratio)||Math.abs(ratio-1)>maxScaleChange)return"scale-discontinuity";if(lock.calibrationCenter&&vlen(v3(frame.center,lock.calibrationCenter))/Math.max(1,lock.calibrationScale)>maxCenterShift)return"position-discontinuity";if(lock.calibrationAxes&&frame.axes.some((axis,i)=>Math.abs(vdot(axis,lock.calibrationAxes[i]))<minAxisDot))return"rotation-discontinuity";return null}
export function binocularRayConfidence(left,right,{minDot=.82}={}){if(!left||!right)return 0;const dot=vdot(left,right);return Number.isFinite(dot)&&dot>=minDot?Math.max(0,Math.min(1,(dot-minDot)/(1-minDot))):0}
export function exponentialSmoothingAlpha(deltaMs,timeConstantMs=95){return 1-Math.exp(-Math.max(0,Math.min(250,deltaMs))/Math.max(1,timeConstantMs))}
export function updatePseudo3DGaze(landmarks,width,height,frame,lock,previous=null,now=globalThis.performance?.now?.()??Date.now(),{timeConstantMs=95,minBinocularDot=.82}={}){if(!frame||!lock||!landmarks?.[468]||!landmarks?.[473]||frame.scale<=1e-6||lock.calibrationScale<=1e-6)return null;const ratio=frame.scale/lock.calibrationScale;if(!Number.isFinite(ratio)||ratio<.55||ratio>1.8)return null;const sphere=o=>vadd(frame.center,matVec(frame.matrix,vmul(o,ratio))),iris=i=>({x:landmarks[i].x*width,y:landmarks[i].y*height,z:landmarks[i].z*width}),leftCenter=sphere(lock.leftOffset),rightCenter=sphere(lock.rightOffset),left=vnorm(v3(iris(468),leftCenter)),right=vnorm(v3(iris(473),rightCenter));if(!left||!right||left.z>=-.08||right.z>=-.08)return null;const binocularDot=vdot(left,right),agreement=binocularRayConfidence(left,right,{minDot:minBinocularDot});if(agreement<=0)return null;let combined=vnorm(vadd(left,right));if(!combined||combined.z>-.08)return null;const dt=Math.max(0,now-(previous?.at??now)),alpha=previous?exponentialSmoothingAlpha(dt,timeConstantMs):1;if(previous?.direction){combined=vnorm(vadd(vmul(previous.direction,1-alpha),vmul(combined,alpha)))}const yaw=-Math.atan2(combined.x,-combined.z)*180/Math.PI,pitch=-Math.atan2(combined.y,-combined.z)*180/Math.PI,confidence=agreement*Math.max(0,1-Math.abs(ratio-1)/.8);return{left,right,direction:combined,leftCenter,rightCenter,scaleRatio:ratio,binocularDot,yaw,pitch,confidence,at:now,source:"pseudo-3d-eye-sphere"}}
export function pseudo3DRaySlope(direction){if(!direction||![direction.x,direction.y,direction.z].every(Number.isFinite)||direction.z>=-.08)return null;return{x:direction.x/-direction.z,y:direction.y/-direction.z}}
export function projectPythonGazeRay(sphereCenter,direction,videoWidth,videoHeight,canvasWidth,canvasHeight,{gazeLength=130,focalScale=.72,maxLength=145,deadZoneDeg=.75}={}){if(!sphereCenter||!direction||![sphereCenter.x,sphereCenter.y,sphereCenter.z,direction.x,direction.y,direction.z,videoWidth,videoHeight,canvasWidth,canvasHeight,gazeLength,focalScale,maxLength,deadZoneDeg].every(Number.isFinite)||videoWidth<=0||videoHeight<=0||canvasWidth<=0||canvasHeight<=0||gazeLength<=0||focalScale<=0||maxLength<=0)return null;const norm=Math.hypot(direction.x,direction.y,direction.z);if(norm<1e-9)return null;const unit={x:direction.x/norm,y:direction.y/norm,z:direction.z/norm},sx=canvasWidth/videoWidth,sy=canvasHeight/videoHeight,start={x:sphereCenter.x*sx,y:sphereCenter.y*sy},forward=Math.max(.08,-unit.z),yaw=Math.atan2(unit.x,forward),pitch=Math.atan2(unit.y,forward),angle=Math.hypot(yaw,pitch),deadZone=deadZoneDeg*Math.PI/180;if(angle<=deadZone)return{start,end:{...start},direction:unit,gazeLength,neutral:true,yaw,pitch,source:"python-eye-sphere-perspective"};const focal=Math.min(canvasWidth,canvasHeight)*focalScale,rawDx=Math.tan(yaw)*focal,rawDy=Math.tan(pitch)*focal,rawLength=Math.hypot(rawDx,rawDy),scale=Math.min(1,maxLength/Math.max(1e-9,rawLength)),end={x:start.x+rawDx*scale,y:start.y+rawDy*scale};return{start,end,direction:unit,gazeLength,neutral:false,yaw,pitch,source:"python-eye-sphere-perspective"}}
export function pseudo3DDisplayTrace(direction,start,eyeSpanPx,{gain=4.5,minLength=58,maxLength=138,deadZone=.012}={}){const slope=pseudo3DRaySlope(direction);if(!slope||!start||![start.x,start.y,eyeSpanPx].every(Number.isFinite)||eyeSpanPx<=0)return null;const sx=Math.max(-1.15,Math.min(1.15,slope.x*gain)),sy=Math.max(-1.15,Math.min(1.15,slope.y*gain)),magnitude=Math.hypot(sx,sy);if(magnitude<=deadZone)return{start:{x:start.x,y:start.y},end:{x:start.x,y:start.y},slope,display:{x:0,y:0},length:0,neutral:true};const authority=Math.min(1,magnitude),base=Math.max(minLength,Math.min(maxLength,eyeSpanPx*4)),length=Math.max(minLength,Math.min(maxLength,base*(.72+.28*authority))),ux=sx/magnitude,uy=sy/magnitude;return{start:{x:start.x,y:start.y},end:{x:start.x+ux*length,y:start.y+uy*length},slope,display:{x:sx,y:sy},length,neutral:false}}
export function pseudo3DTraceVisualStyle(gaze,{minConfidence=.35}={}){const confidence=Math.max(0,Math.min(1,Number(gaze?.confidence)||0)),age=Math.max(0,(Number(gaze?.now)||Number(gaze?.at)||0)-Number(gaze?.at||0));if(confidence<minConfidence||age>260)return null;const freshness=Math.max(.25,1-age/260),authority=confidence*freshness;return{opacity:.28+.72*authority,lineWidth:.8+1.5*authority,lengthScale:.58+.42*authority,dash:authority<.58?[5,4]:[]}}

export function eyeMarkerStyle(source){
  const measured=source==="pupil";
  return measured?{source:"pupil",label:"MEASURED HYBRID",color:"rgba(255,42,61,.98)",dash:[],measured:true}:{source:"landmark",label:"LANDMARK FALLBACK",color:"rgba(255,177,74,.98)",dash:[4,3],measured:false};
}

function signedDeadZone(value,deadZone,limit){const magnitude=Math.abs(value);return magnitude<=deadZone?0:Math.sign(value)*Math.min(1,(magnitude-deadZone)/Math.max(.001,limit-deadZone))}

export function normalizeEyeAlignment(feature,{deadZoneX=.025,deadZoneY=.04,limitX=.34,limitY=.7}={}){
  if(!feature||feature.source==="face-flow"||![feature.x,feature.y].every(Number.isFinite))return null;
  return{x:signedDeadZone(feature.x,deadZoneX,limitX),y:signedDeadZone(feature.y,deadZoneY,limitY),confidence:Math.max(0,Math.min(1,Number(feature.pupilConfidence??feature.confidence) || 0)),source:feature.source||"landmark"};
}

export function updateEyeAlignmentState(previous,current,now,{holdMs=140,easeMs=220}={}){
  if(!Number.isFinite(now))return previous||null;
  if(!current){if(!previous)return null;const age=now-(previous.lastSeenAt??previous.at??now);if(age<=holdMs)return{...previous,at:now,stale:true};const ease=Math.min(1,(age-holdMs)/easeMs),keep=1-ease;if(keep<=0)return null;return{...previous,x:previous.x*keep,y:previous.y*keep,confidence:previous.confidence*keep,at:now,stale:true}}
  const confidence=Math.max(0,Math.min(1,current.confidence??0)),sourceSwitch=Boolean(previous&&previous.source!==current.source),alpha=previous?Math.min(.78,Math.max(.14,(sourceSwitch?.18:.28)+confidence*.42)):1;
  return{x:previous?previous.x+(current.x-previous.x)*alpha:current.x,y:previous?previous.y+(current.y-previous.y)*alpha:current.y,confidence,source:current.source,at:now,lastSeenAt:now,stale:false};
}

export function eyeAlignmentTrace(center,feature,eyeSpan,width,height,{maxAngleDeg=34,minLength=52,maxLength=150,basis=null,normalized=false}={}){
  const alignment=normalized?feature:normalizeEyeAlignment(feature);
  if(!center||!alignment||![center.x,center.y,alignment.x,alignment.y,eyeSpan,width,height].every(Number.isFinite)||eyeSpan<=0||width<=0||height<=0)return null;
  const start={x:center.x*width,y:center.y*height};
  let hx=1,hy=0,spanPx=Math.max(8,eyeSpan*width);
  if(basis?.a&&basis?.b&&[basis.a.x,basis.a.y,basis.b.x,basis.b.y].every(Number.isFinite)){
    const dx=(basis.b.x-basis.a.x)*width,dy=(basis.b.y-basis.a.y)*height,norm=Math.hypot(dx,dy);
    if(norm>1){hx=dx/norm;hy=dy/norm;spanPx=Math.max(8,norm)}
  }
  let vx=-hy,vy=hx;if(vy<0){vx=-vx;vy=-vy}
  // Keep headroom at both caps so normal eye motion can change the tip instead of
  // pinning it. X steers across the local eye axis and Y changes its down component.
  const nominal=Math.max(minLength*1.18,Math.min(maxLength*.74,spanPx*3.65)),lateral=nominal*Math.tan(maxAngleDeg*Math.PI/180)*alignment.x,forward=nominal*(1+alignment.y*.75),vectorX=hx*lateral+vx*forward,vectorY=hy*lateral+vy*forward,rawLength=Math.hypot(vectorX,vectorY),length=Math.max(minLength,Math.min(maxLength,rawLength)),scale=length/Math.max(.001,rawLength),dx=vectorX*scale,dy=vectorY*scale,endX=start.x+dx,endY=start.y+dy;
  return{start,end:{x:Math.max(0,Math.min(width,endX)),y:Math.max(0,Math.min(height,endY))},length,angleDeg:Math.atan2(lateral,forward)*180/Math.PI,alignment:{x:alignment.x,y:alignment.y},confidence:alignment.confidence,stale:Boolean(alignment.stale),source:alignment.source||"landmark",basis:{horizontal:{x:hx,y:hy},vertical:{x:vx,y:vy},spanPx}};
}

export function eyeTraceVisualStyle(trace){
  const confidence=Math.max(0,Math.min(1,Number(trace?.confidence)||0)),fallback=trace?.source!=="pupil",stale=Boolean(trace?.stale),authority=(.3+.7*confidence)*(fallback?.72:1)*(stale?.55:1);
  return{opacity:.22+.78*authority,lineWidth:.7+1.15*authority,dash:stale?[2,5]:fallback?[4,4]:[6,3],lengthScale:.62+.38*authority};
}

export function binocularAlignmentsDisagree(left,right,{limit=.72}={}){
  return Boolean(left&&right&&Math.hypot(left.x-right.x,left.y-right.y)>limit);
}

export function personFramingFromLandmarks(marks,{targetFaceWidth=.34,maxScale=1.65}={}){
  const points=(marks||[]).filter(point=>Number.isFinite(point?.x)&&Number.isFinite(point?.y));
  if(points.length<2)return null;
  const xs=points.map(point=>point.x),ys=points.map(point=>point.y),minX=Math.min(...xs),maxX=Math.max(...xs),maxY=Math.max(...ys);
  const faceWidth=Math.max(.01,maxX-minX),x=Math.max(.2,Math.min(.8,(minX+maxX)/2)),y=Math.max(.38,Math.min(.74,maxY+.2));
  return{scale:Math.max(1,Math.min(maxScale,targetFaceWidth/faceWidth)),x,y};
}

export function rgbaToLuminance(data,width,height,{step=1}={}){
  if(!data||!width||!height||data.length<width*height*4)return null;
  const outWidth=Math.ceil(width/step),outHeight=Math.ceil(height/step),luma=new Uint8ClampedArray(outWidth*outHeight);
  let n=0;for(let y=0;y<height;y+=step)for(let x=0;x<width;x+=step){const i=(y*width+x)*4;luma[n++]=Math.round(.299*data[i]+.587*data[i+1]+.114*data[i+2])}
  return{data:luma,width:outWidth,height:outHeight};
}

export function extractLuminancePatch(frame,{x=0,y=0,w=1,h=1},outWidth,outHeight){
  if(!frame?.data||!frame.width||!frame.height||!outWidth||!outHeight)return null;
  const rgba=new Uint8ClampedArray(outWidth*outHeight*4),left=Math.max(0,Math.min(frame.width-1,Math.floor(x*frame.width))),top=Math.max(0,Math.min(frame.height-1,Math.floor(y*frame.height))),right=Math.max(left+1,Math.min(frame.width,Math.ceil((x+w)*frame.width))),bottom=Math.max(top+1,Math.min(frame.height,Math.ceil((y+h)*frame.height)));
  for(let oy=0;oy<outHeight;oy++)for(let ox=0;ox<outWidth;ox++){const sx=Math.min(right-1,left+Math.floor((ox+.5)*(right-left)/outWidth)),sy=Math.min(bottom-1,top+Math.floor((oy+.5)*(bottom-top)/outHeight)),v=frame.data[sy*frame.width+sx],i=(oy*outWidth+ox)*4;rgba[i]=rgba[i+1]=rgba[i+2]=v;rgba[i+3]=255}
  return{data:rgba,width:outWidth,height:outHeight};
}

export function propagateLandmarks(landmarks,flow,width,height,ageMs,{ttlMs=250,minConfidence=.12}={}){
  if(!Array.isArray(landmarks)||!landmarks.length||!flow||ageMs<0||ageMs>ttlMs||flow.confidence<minConfidence||!width||!height)return null;
  const decay=Math.max(0,1-ageMs/ttlMs),dx=flow.dx/width,dy=flow.dy/height;
  return{landmarks:landmarks.map(p=>({...p,x:p.x+dx,y:p.y+dy})),confidence:flow.confidence*decay,source:"face-flow",ageMs};
}

export function observationMayBeAccepted(observation){return Boolean(observation&&observation.source!=="face-flow"&&observation.synthetic!==true)}

export function estimateLuminanceTranslation(previous,current,width,height,{maxShift=4,sampleStep=2,roi=null}={}){
  if(!previous||!current||previous.length!==current.length||previous.length<width*height)return null;
  const left=Math.max(maxShift,Math.floor((roi?.x??0)*width)),right=Math.min(width-maxShift,Math.ceil(((roi?.x??0)+(roi?.w??1))*width)),top=Math.max(maxShift,Math.floor((roi?.y??0)*height)),bottom=Math.min(height-maxShift,Math.ceil(((roi?.y??0)+(roi?.h??1))*height));if(right-left<8||bottom-top<8)return null;
  let best=null,zeroError=0,scores=[],texture=0,textureCount=0;
  for(let y=top;y<bottom;y+=sampleStep)for(let x=left;x<right;x+=sampleStep){texture+=Math.abs(previous[y*width+x]-previous[y*width+Math.min(width-1,x+1)]);textureCount++}
  for(let dy=-maxShift;dy<=maxShift;dy++)for(let dx=-maxShift;dx<=maxShift;dx++){
    let error=0,count=0;for(let y=top;y<bottom;y+=sampleStep)for(let x=left;x<right;x+=sampleStep){error+=Math.abs(previous[y*width+x]-current[(y+dy)*width+x+dx]);count++}
    const mean=error/Math.max(1,count),entry={dx,dy,error:mean};scores.push(entry);if(dx===0&&dy===0)zeroError=mean;if(!best||mean<best.error)best=entry;
  }
  if(!best)return null;scores.sort((a,b)=>a.error-b.error);const second=scores.find(s=>Math.abs(s.dx-best.dx)>1||Math.abs(s.dy-best.dy)>1)||scores[1],ambiguity=Math.max(0,(second.error-best.error)/(second.error+1)),textureScore=Math.min(1,(texture/Math.max(1,textureCount))/12),reverse=estimateTranslationOneWay(current,previous,width,height,{maxShift,sampleStep,roi}),fb=reverse?Math.hypot(best.dx+reverse.dx,best.dy+reverse.dy):99,consistency=Math.max(0,1-fb/2),gain=Math.max(0,(zeroError-best.error)/(zeroError+1)),confidence=Math.min(1,gain*.35+ambiguity*.25+textureScore*.2+consistency*.2);return{...best,confidence,ambiguity,texture:textureScore,forwardBackwardError:fb};
}

function estimateTranslationOneWay(previous,current,width,height,{maxShift,sampleStep,roi}){const left=Math.max(maxShift,Math.floor((roi?.x??0)*width)),right=Math.min(width-maxShift,Math.ceil(((roi?.x??0)+(roi?.w??1))*width)),top=Math.max(maxShift,Math.floor((roi?.y??0)*height)),bottom=Math.min(height-maxShift,Math.ceil(((roi?.y??0)+(roi?.h??1))*height));let best=null;for(let dy=-maxShift;dy<=maxShift;dy++)for(let dx=-maxShift;dx<=maxShift;dx++){let error=0,count=0;for(let y=top;y<bottom;y+=sampleStep)for(let x=left;x<right;x+=sampleStep){error+=Math.abs(previous[y*width+x]-current[(y+dy)*width+x+dx]);count++}const mean=error/Math.max(1,count);if(!best||mean<best.error)best={dx,dy,error:mean}}return best}

export function selectHeadKeyframe(keyframes,head,{maxYaw=26,maxPitch=22,maxRoll=20,maxZ=3}={}){
  if(!head||!(keyframes||[]).length)return null;let best=null;
  for(const keyframe of keyframes){const delta={yaw:head.yaw-keyframe.head.yaw,pitch:head.pitch-keyframe.head.pitch,roll:head.roll-keyframe.head.roll,z:head.z-keyframe.head.z},score=Math.hypot(delta.yaw/maxYaw,delta.pitch/maxPitch,delta.roll/maxRoll,delta.z/maxZ);if(!best||score<best.score)best={keyframe,delta,score,compatible:Math.abs(delta.yaw)<=maxYaw&&Math.abs(delta.pitch)<=maxPitch&&Math.abs(delta.roll)<=maxRoll&&Math.abs(delta.z)<=maxZ}}
  return best;
}

export function addHeadKeyframe(keyframes,head,{limit=5,minSeparation=.18,now=Date.now(),eyes=null,source="manual",quality=1}={}){
  if(!head||![head.yaw,head.pitch,head.roll,head.x,head.y,head.z].every(Number.isFinite))return keyframes||[];
  const frames=validateHeadKeyframes(keyframes),candidate={id:`head-${now}`,createdAt:new Date(now).toISOString(),head:{...head},eyes:eyes?structuredClone(eyes):null,source,quality:Math.max(0,Math.min(1,quality))};
  const nearest=selectHeadKeyframe(frames,head);if(nearest&&nearest.score<minSeparation)return frames;
  frames.push(candidate);return frames.slice(-limit);
}

export function validateHeadKeyframes(keyframes){return (Array.isArray(keyframes)?keyframes:[]).filter(k=>k&&typeof k.id==="string"&&k.head&&[k.head.yaw,k.head.pitch,k.head.roll,k.head.x,k.head.y,k.head.z].every(Number.isFinite)&&(!k.eyes||[k.eyes.left?.x,k.eyes.left?.y,k.eyes.right?.x,k.eyes.right?.y].every(Number.isFinite))).map(k=>({...k,quality:Number.isFinite(k.quality)?Math.max(0,Math.min(1,k.quality)):1,source:typeof k.source==="string"?k.source:"restored"}))}

export function parseHeadRaw(text,{scale=.01}={}){
  const triangles=[],lines=String(text||"").split(/\r?\n/).filter(line=>line.trim());for(const line of lines){const values=line.trim().split(/\s+/).map(Number);if(values.length!==9||!values.every(Number.isFinite))return{triangleCount:0,bounds:null,centroid:null,valid:false,error:"Malformed head.raw triangle row"};triangles.push(values.map(value=>value*scale))}
  if(!triangles.length)return{triangleCount:0,bounds:null,centroid:null,valid:false,error:"Empty head.raw"};const vertices=triangles.flatMap(t=>[[t[0],t[1],t[2]],[t[3],t[4],t[5]],[t[6],t[7],t[8]]]);const axis=i=>vertices.map(v=>v[i]),xs=axis(0),ys=axis(1),zs=axis(2),bounds={min:[Math.min(...xs),Math.min(...ys),Math.min(...zs)],max:[Math.max(...xs),Math.max(...ys),Math.max(...zs)]};return{triangleCount:triangles.length,bounds,centroid:[xs.reduce((a,b)=>a+b,0)/xs.length,ys.reduce((a,b)=>a+b,0)/ys.length,zs.reduce((a,b)=>a+b,0)/zs.length],valid:true,canonicalAspect:(bounds.max[0]-bounds.min[0])/Math.max(.001,bounds.max[1]-bounds.min[1])};
}

export function landmarkGeometrySample(landmarks,cameraAspect=1){if(!Array.isArray(landmarks)||landmarks.length<2||!Number.isFinite(cameraAspect)||cameraAspect<=0)return null;const points=landmarks.slice(0,468);if(!points.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)))return null;const xs=points.map(p=>p.x),ys=points.map(p=>p.y),w=(Math.max(...xs)-Math.min(...xs))*cameraAspect,h=Math.max(...ys)-Math.min(...ys);return w>.001&&h>.001?{canonicalAspect:w/h,cameraAspect}:null}
export function createUserGeometryProfile(samples){const valid=(samples||[]).filter(s=>Number.isFinite(s.canonicalAspect)&&s.canonicalAspect>.1&&s.canonicalAspect<4);if(valid.length<10)return null;const values=valid.map(s=>s.canonicalAspect).sort((a,b)=>a-b),center=values[Math.floor(values.length/2)],inliers=valid.filter(s=>Math.abs(Math.log(s.canonicalAspect/center))<.08);if(inliers.length<10)return null;const sorted=inliers.map(s=>s.canonicalAspect).sort((a,b)=>a-b);return{valid:true,source:"user-calibration-landmarks",canonicalAspect:sorted[Math.floor(sorted.length/2)],cameraAspect:inliers[0].cameraAspect,sampleCount:inliers.length,createdAt:new Date().toISOString()}}
export function headGeometryRegularizer(geometry,landmarks,cameraAspect=1){if(!geometry?.valid||!Number.isFinite(geometry.canonicalAspect))return null;const sample=landmarkGeometrySample(landmarks,cameraAspect);if(!sample||geometry.cameraAspect&&Math.abs(geometry.cameraAspect-cameraAspect)>.01)return null;const aspect=sample.canonicalAspect,deviation=Math.abs(Math.log(aspect/geometry.canonicalAspect)),weight=Math.max(0,Math.min(.18,.18*Math.exp(-deviation)));return{aspect,deviation,weight,source:"user-calibration-landmarks"}}

export function calibrationFeatures(observation,mode=9){
  if(!observation?.leftEye||!observation?.rightEye||!observation?.head)return null;
  const full=[1,observation.leftEye.x,observation.leftEye.y,observation.rightEye.x,observation.rightEye.y,observation.head.yaw,observation.head.pitch];
  const compact=[1,(observation.leftEye.x+observation.rightEye.x)/2,(observation.leftEye.y+observation.rightEye.y)/2,observation.head.yaw,observation.head.pitch];
  const values=mode===5?compact:full;
  return values.every(Number.isFinite)?values:null;
}

export function analyzePupil(imageData,{darkQuantile=.3,minArea=.025,maxArea=.42}={}){
  const {data,width,height}=imageData||{};
  if(!data||!width||!height||data.length<width*height*4)return null;
  const pixels=[],lum=new Float32Array(width*height);
  for(let y=1;y<height-1;y++)for(let x=1;x<width-1;x++){
    const i=(y*width+x)*4;
    const l=.299*data[i]+.587*data[i+1]+.114*data[i+2];lum[y*width+x]=l;pixels.push({x,y,l});
  }
  if(!pixels.length)return null;
  const sorted=pixels.map(p=>p.l).sort((a,b)=>a-b),cut=sorted[Math.floor((sorted.length-1)*darkQuantile)],bright=sorted[Math.floor((sorted.length-1)*.72)];
  const threshold=Math.min(cut,(sorted[0]+bright)*.5),seen=new Uint8Array(width*height),components=[];
  for(let y=1;y<height-1;y++)for(let x=1;x<width-1;x++){
    const seed=y*width+x,nx=(x-(width-1)/2)/((width-2)/2),ny=(y-(height-1)/2)/((height-2)/2);
    if(seen[seed]||lum[seed]>threshold||nx*nx+ny*ny/.52**2>1)continue;
    const queue=[seed],component=[];seen[seed]=1;
    while(queue.length){const index=queue.pop(),px=index%width,py=Math.floor(index/width);component.push(index);for(const d of [-1,1,-width,width]){const ni=index+d,qx=ni%width,qy=Math.floor(ni/width),ex=(qx-(width-1)/2)/((width-2)/2),ey=(qy-(height-1)/2)/((height-2)/2);if(qx<1||qx>=width-1||qy<1||qy>=height-1||seen[ni]||lum[ni]>threshold||ex*ex+ey*ey/.52**2>1)continue;seen[ni]=1;queue.push(ni)}}
    components.push(component);
  }
  let best=null;
  for(const component of components){const area=component.length/pixels.length;if(area<minArea||area>maxArea)continue;let sx=0,sy=0,sw=0,sumLum=0,minX=width,maxX=0,minY=height,maxY=0;for(const index of component){const x=index%width,y=Math.floor(index/width),w=threshold-lum[index]+1;sx+=x*w;sy+=y*w;sw+=w;sumLum+=lum[index];minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y)}const x=sx/sw,y=sy/sw,aspect=(maxX-minX+1)/(maxY-minY+1),shape=Math.exp(-Math.abs(Math.log(Math.max(.1,aspect)))),eccentricity=Math.hypot(x/(width-1)-.5,(y/(height-1)-.5)*1.5),center=Math.max(0,1-eccentricity/.72),meanLum=sumLum/component.length,contrast=Math.max(0,bright-meanLum)/255,borderContact=minX<=1||maxX>=width-2||minY<=1||maxY>=height-2,score=contrast*.45+shape*.25+center*.2+Math.min(1,area/.12)*.1-(borderContact?.3:0)-Math.max(0,eccentricity-.55)*.5,confidence=Math.max(0,Math.min(1,(score-.06)/.58));if(!best||score>best.score)best={x:x/(width-1),y:y/(height-1),confidence,contrast,coverage:area,score,aspect,meanLuminance:meanLum,borderContact,eccentricity}}
  return best;
}

export function hybridCoverageReport(samples,{minEyeCoverage=.65,minBinocularCoverage=.55,minConfidence=.28,maxSwitchRate=.4}={}){
  const groups=new Map();
  for(const sample of samples||[]){const key=sample.targetIndex??"validation",o=sample.observation||sample;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(o)}
  const targets=[...groups.entries()].map(([targetIndex,items])=>{const metric=side=>{let pupil=0,confidence=0,switches=0,previous="";for(const o of items){const eye=o?.[`${side}Eye`]||{},source=eye.source||"landmark",active=source==="pupil";if(active){pupil++;confidence+=Number(eye.pupilConfidence)||Number(o.hybridConfidence)||0}if(previous&&source!==previous)switches++;previous=source}return{coverage:items.length?pupil/items.length:0,meanConfidence:pupil?confidence/pupil:0,switchRate:items.length>1?switches/(items.length-1):0,fallbackRate:items.length?1-pupil/items.length:1}};const left=metric("left"),right=metric("right");let binocular=0;for(const o of items)if(o?.leftEye?.source==="pupil"&&o?.rightEye?.source==="pupil")binocular++;const binocularCoverage=items.length?binocular/items.length:0,passed=left.coverage>=minEyeCoverage&&right.coverage>=minEyeCoverage&&binocularCoverage>=minBinocularCoverage&&left.meanConfidence>=minConfidence&&right.meanConfidence>=minConfidence&&left.switchRate<=maxSwitchRate&&right.switchRate<=maxSwitchRate;return{targetIndex,count:items.length,left,right,binocularCoverage,passed}});
  const aggregate=side=>({coverage:targets.length?Math.min(...targets.map(t=>t[side].coverage)):0,meanConfidence:targets.length?targets.reduce((s,t)=>s+t[side].meanConfidence,0)/targets.length:0,maxSwitchRate:targets.length?Math.max(...targets.map(t=>t[side].switchRate)):0,maxFallbackRate:targets.length?Math.max(...targets.map(t=>t[side].fallbackRate)):1});
  const left=aggregate("left"),right=aggregate("right"),binocularCoverage=targets.length?Math.min(...targets.map(t=>t.binocularCoverage)):0;return{passed:targets.length>0&&targets.every(t=>t.passed),targets,left,right,binocularCoverage,coverage:Math.min(left.coverage,right.coverage),meanConfidence:(left.meanConfidence+right.meanConfidence)/2,maxSwitchRate:Math.max(left.maxSwitchRate,right.maxSwitchRate),thresholds:{minEyeCoverage,minBinocularCoverage,minConfidence,maxSwitchRate}};
}

export function estimatePatchTranslation(previous,current,width,height,{maxShift=3}={}){
  if(!previous||!current||previous.length!==current.length||previous.length<width*height*4)return null;const a=rgbaToLuminance(previous,width,height),b=rgbaToLuminance(current,width,height),flow=estimateLuminanceTranslation(a.data,b.data,width,height,{maxShift,sampleStep:1});return flow?{...flow,pixelDx:flow.dx,pixelDy:flow.dy,dx:flow.dx/(width-1),dy:flow.dy/(height-1)}:null;
}

export function trackPupil(previous,current,{alpha=.42,maxJump=.34,minConfidence=.12}={}){
  if(!current||current.confidence<minConfidence)return{value:null,reason:"low-confidence"};
  if(!previous)return{value:{...current},reason:"acquired"};
  const jump=Math.hypot(current.x-previous.x,current.y-previous.y),shapeDelta=Math.abs(Math.log(Math.max(.1,current.aspect||1)/Math.max(.1,previous.aspect||1)));
  if(shapeDelta>.65)return{value:null,reason:"shape-instability"};
  if(jump>maxJump)return{value:null,reason:"jump"};
  const a=Math.max(.18,Math.min(.75,alpha*current.confidence+.16));
  return{value:{...current,x:previous.x+(current.x-previous.x)*a,y:previous.y+(current.y-previous.y)*a,confidence:Math.min(current.confidence,previous.confidence*.7+current.confidence*.3)},reason:"tracked"};
}

export function applyCalibrationSnapshot(model,snapshot){
  if(!model||!snapshot)return false;
  for(const key of ["featureMode","modelX","modelY","leftX","leftY","rightX","rightY","eyeResiduals","headRef"]){
    model[key]=snapshot[key]===undefined?null:structuredClone(snapshot[key]);
  }
  model.previous=null;
  return true;
}

export function calibrationRecordIsValid(record,{maxMedian=.08,maxP95=.16}={}){
  const mode=record?.featureMode||9,expected=mode===5?5:7,models=[record?.modelX,record?.modelY,record?.leftX,record?.leftY,record?.rightX,record?.rightY];
  return record?.schema===4&&record?.meta?.validation?.accepted===true&&record.meta.validation.median<=maxMedian&&record.meta.validation.p95<=maxP95&&models.every(model=>Array.isArray(model)&&model.length===(model===record.modelX||model===record.modelY?expected:5)&&model.every(Number.isFinite));
}
