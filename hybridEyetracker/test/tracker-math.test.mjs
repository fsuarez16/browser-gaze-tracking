import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import {adaptiveDriftFeatures,addHeadKeyframe,analyzePupil,applyCalibrationSnapshot,binocularAlignmentsDisagree,binocularRayConfidence,calibrationFeatures,calibrationRecordIsValid,createAdaptiveDriftModel,diagonalError,estimateLuminanceTranslation,estimatePatchTranslation,exponentialSmoothingAlpha,extractLuminancePatch,eyeAlignmentTrace,eyeMarkerStyle,eyeSphereLockInvalidationReason,eyeTraceVisualStyle,flushGazeEventClassifier,gazeBubbleGate,headGeometryRegularizer,hybridCoverageReport,lockEyeSpheres,meanPairwiseScale,normalizeEyeAlignment,nosePcaFrame,observationMayBeAccepted,parseHeadRaw,personFramingFromLandmarks,pixelError,predictAdaptiveDrift,projectPythonGazeRay,propagateLandmarks,pseudo3DDisplayTrace,pseudo3DRaySlope,pseudo3DTraceVisualStyle,PYTHON_NOSE_INDICES,rgbaToLuminance,selectHeadKeyframe,symmetricEigen3,trackPupil,updateAdaptiveDriftRls,updateEyeAlignmentState,updateGazeEventClassifier,updatePseudo3DGaze,validateAdaptiveDriftModel,validateHeadKeyframes} from "../dist/tracker-math.js";
import {updateOneEuro2D} from "../dist/tracker-math.js";

test("gaze bubble holds through a brief rejected frame and hides after sustained loss",()=>{
  const base={enabled:true,tracking:true,hasPoint:true,lastValidAt:1000,reason:"low-confidence"};
  assert.deepEqual(gazeBubbleGate({...base,now:1120}),{visible:true,holding:true,age:120});
  assert.deepEqual(gazeBubbleGate({...base,now:1400}),{visible:false,holding:false,age:400});
  assert.equal(gazeBubbleGate({...base,now:1050,reason:"not-calibrated"}).visible,false);
  assert.equal(gazeBubbleGate({...base,now:1050,enabled:false}).visible,false);
});

test("One Euro gaze filter damps jitter, follows motion and resets after a gap",()=>{
  let sample=updateOneEuro2D(null,{x:.5,y:.5},0);assert.deepEqual(sample.point,{x:.5,y:.5});
  sample=updateOneEuro2D(sample.state,{x:.51,y:.49},16,{confidence:.8});assert.ok(sample.point.x>.5&&sample.point.x<.51);assert.ok(sample.point.y<.5&&sample.point.y>.49);
  const moving=updateOneEuro2D(sample.state,{x:.8,y:.2},32,{confidence:1});assert.ok(moving.point.x>sample.point.x&&moving.point.y<sample.point.y);
  const reset=updateOneEuro2D(moving.state,{x:.2,y:.7},250,{maxGapMs:150});assert.deepEqual(reset.point,{x:.2,y:.7});
});

test("pixel error scales each normalized axis independently",()=>{
  assert.equal(pixelError(.1,.1,1000,500),Math.hypot(100,50));
  assert.notEqual(pixelError(.1,.1,1000,500),Math.hypot(.1,.1)*Math.hypot(1000,500));
});

test("diagonal error is pixel error divided by viewport diagonal",()=>{
  assert.equal(diagonalError(.1,.1,1000,500),Math.hypot(100,50)/Math.hypot(1000,500));
});

test("eye alignment trace starts at the measured center and extends downward",()=>{
  const trace=eyeAlignmentTrace({x:.5,y:.25},{x:.2,y:-.1,source:"pupil"},.1,1000,500);
  assert.deepEqual(trace.start,{x:500,y:125});
  assert.ok(trace.end.y>trace.start.y);
  assert.ok(trace.end.x>trace.start.x);
  assert.equal(trace.source,"pupil");
});

test("eye alignment endpoints visibly separate for left and right pupil motion",()=>{
  const left=eyeAlignmentTrace({x:.5,y:.25},{x:-.25,y:0,confidence:1,source:"pupil"},.1,1000,500);
  const right=eyeAlignmentTrace({x:.5,y:.25},{x:.25,y:0,confidence:1,source:"pupil"},.1,1000,500);
  assert.ok(left.end.x<450&&right.end.x>550);
  assert.ok(right.end.x-left.end.x>100);
});

test("centered alignment follows local down and dead zones tiny jitter",()=>{
  assert.deepEqual(normalizeEyeAlignment({x:.02,y:-.03,confidence:.8,source:"pupil"}),{x:0,y:0,confidence:.8,source:"pupil"});
  const trace=eyeAlignmentTrace({x:.5,y:.25},{x:0,y:0,confidence:1,source:"pupil"},.1,1000,500);
  assert.equal(trace.end.x,trace.start.x);assert.ok(trace.end.y>trace.start.y);
});

test("vertical eye alignment changes trace length within safe bounds",()=>{
  const up=eyeAlignmentTrace({x:.5,y:.25},{x:0,y:-.6,confidence:1,source:"pupil"},.1,1000,500);
  const down=eyeAlignmentTrace({x:.5,y:.25},{x:0,y:.6,confidence:1,source:"pupil"},.1,1000,500);
  assert.ok(down.length>up.length);assert.ok(up.length>=52&&down.length<=150);
});

test("alignment smoothing blends source switches and briefly holds stale input",()=>{
  const initial=updateEyeAlignmentState(null,{x:1,y:0,confidence:1,source:"pupil"},1000);
  const switched=updateEyeAlignmentState(initial,{x:-1,y:0,confidence:.5,source:"landmark"},1016);
  assert.ok(switched.x>0&&switched.x<1);
  const held=updateEyeAlignmentState(switched,null,1100);assert.equal(held.x,switched.x);assert.equal(held.stale,true);
  const eased=updateEyeAlignmentState(switched,null,1300);assert.ok(Math.abs(eased.x)<Math.abs(switched.x));
  assert.equal(normalizeEyeAlignment({x:.2,y:0,source:"face-flow"}),null);
});

test("small realistic 2D eye motion remains visible after multi-frame settling",()=>{
  let up=null,down=null;
  for(let frame=0;frame<12;frame++){
    up=updateEyeAlignmentState(up,normalizeEyeAlignment({x:.04,y:-.06,confidence:.9,source:"pupil"}),1000+frame*16);
    down=updateEyeAlignmentState(down,normalizeEyeAlignment({x:.04,y:.06,confidence:.9,source:"pupil"}),1000+frame*16);
  }
  const upTrace=eyeAlignmentTrace({x:.5,y:.3},up,.1,1000,600,{normalized:true});
  const downTrace=eyeAlignmentTrace({x:.5,y:.3},down,.1,1000,600,{normalized:true});
  assert.ok(Math.hypot(downTrace.end.x-upTrace.end.x,downTrace.end.y-upTrace.end.y)>3);
  assert.notEqual(downTrace.angleDeg,upTrace.angleDeg);
  assert.ok(upTrace.end.y>upTrace.start.y&&downTrace.end.y>downTrace.start.y);
});

test("small signed horizontal motion produces separated settled tips",()=>{
  let left=null,right=null;
  for(let frame=0;frame<12;frame++){
    left=updateEyeAlignmentState(left,normalizeEyeAlignment({x:-.06,y:.04,confidence:.9,source:"pupil"}),2000+frame*16);
    right=updateEyeAlignmentState(right,normalizeEyeAlignment({x:.06,y:.04,confidence:.9,source:"pupil"}),2000+frame*16);
  }
  const a=eyeAlignmentTrace({x:.5,y:.3},left,.1,1000,600,{normalized:true}),b=eyeAlignmentTrace({x:.5,y:.3},right,.1,1000,600,{normalized:true});
  assert.ok(b.end.x-a.end.x>10);
});

test("trace authority styling weakens fallback and stale estimates",()=>{
  const measured=eyeTraceVisualStyle({confidence:.9,source:"pupil",stale:false}),fallback=eyeTraceVisualStyle({confidence:.25,source:"landmark",stale:false}),stale=eyeTraceVisualStyle({confidence:.25,source:"landmark",stale:true});
  assert.ok(measured.opacity>fallback.opacity&&fallback.opacity>stale.opacity);
  assert.ok(measured.lineWidth>fallback.lineWidth&&fallback.lineWidth>stale.lineWidth);
  assert.ok(measured.lengthScale>fallback.lengthScale&&fallback.lengthScale>stale.lengthScale);
  assert.notDeepEqual(fallback.dash,stale.dash);
});

test("binocular disagreement considers both normalized axes",()=>{
  assert.equal(binocularAlignmentsDisagree({x:0,y:-.5},{x:0,y:.5}),true);
  assert.equal(binocularAlignmentsDisagree({x:.1,y:.1},{x:.12,y:.14}),false);
});

test("eye alignment trace clamps extreme deflection and rejects malformed input",()=>{
  const trace=eyeAlignmentTrace({x:.99,y:.95},{x:99,y:99},.2,100,50);
  assert.ok(trace.end.x<=100);
  assert.ok(trace.end.y<=50);
  assert.equal(eyeAlignmentTrace(null,{x:0,y:0},.2,100,50),null);
});

test("eye alignment trace uses pixel-space eye basis at 4:3 and 16:9",()=>{
  const options={basis:{a:{x:.4,y:.4},b:{x:.6,y:.5}}};
  const fourThree=eyeAlignmentTrace({x:.5,y:.45},{x:.5,y:0,source:"pupil"},.2,640,480,options);
  const wide=eyeAlignmentTrace({x:.5,y:.45},{x:.5,y:0,source:"pupil"},.2,1280,720,options);
  assert.ok(fourThree.basis.horizontal.y>0);
  assert.ok(wide.basis.horizontal.y>0);
  assert.notEqual(fourThree.basis.horizontal.y,wide.basis.horizontal.y);
  assert.ok(fourThree.end.y>fourThree.start.y&&wide.end.y>wide.start.y);
});

test("rolled-eye local deflection follows the local horizontal axis and remains bounded",()=>{
  const trace=eyeAlignmentTrace({x:.5,y:.5},{x:.7,y:0,source:"pupil"},.2,800,600,{basis:{a:{x:.4,y:.35},b:{x:.6,y:.55}}});
  assert.ok(trace.basis.horizontal.y>0);
  assert.ok(trace.end.y>trace.start.y);
  assert.ok(trace.end.x>=0&&trace.end.x<=800&&trace.end.y<=600);
});

test("eye marker style distinguishes measured pupils from landmark fallback",()=>{
  assert.deepEqual(eyeMarkerStyle("pupil").dash,[]);
  assert.equal(eyeMarkerStyle("pupil").label,"MEASURED HYBRID");
  assert.deepEqual(eyeMarkerStyle("landmark").dash,[4,3]);
  assert.equal(eyeMarkerStyle("landmark").label,"LANDMARK FALLBACK");
});

test("person framing centers a small face and preserves upper-body space",()=>{
  const frame=personFramingFromLandmarks([{x:.4,y:.2},{x:.6,y:.5}]);
  assert.equal(frame.scale,1.65);
  assert.equal(frame.x,.5);
  assert.equal(frame.y,.7);
});

test("person framing avoids unnecessary zoom and clamps edge origins",()=>{
  const frame=personFramingFromLandmarks([{x:-.1,y:.1},{x:.5,y:.7}]);
  assert.equal(frame.scale,1);
  assert.equal(frame.x,.2);
  assert.equal(frame.y,.74);
  assert.equal(personFramingFromLandmarks([]),null);
});

test("five-point calibration uses five coefficients",()=>{
  const observation={leftEye:{x:.1,y:.2},rightEye:{x:.3,y:.4},head:{yaw:2,pitch:3}};
  assert.deepEqual(calibrationFeatures(observation,5),[1,.2,.30000000000000004,2,3]);
  assert.equal(calibrationFeatures(observation,9).length,7);
});

test("invalid observations are rejected",()=>{
  assert.equal(calibrationFeatures({leftEye:{x:NaN,y:0},rightEye:{x:0,y:0},head:{yaw:0,pitch:0}},5),null);
});

test("a previous accepted calibration snapshot can be restored after cancellation",()=>{
  const model={featureMode:9,modelX:[9],modelY:[8],previous:{x:.2,y:.3}};
  const accepted={featureMode:5,modelX:[1,2],modelY:[3,4],leftX:[5],leftY:[6],rightX:[7],rightY:[8],eyeResiduals:{left:.1,right:.2},headRef:{yaw:1}};
  assert.equal(applyCalibrationSnapshot(model,accepted),true);
  assert.deepEqual(model.modelX,[1,2]);
  assert.deepEqual(model.eyeResiduals,{left:.1,right:.2});
  assert.equal(model.previous,null);
});

test("missing calibration snapshots are not applied",()=>{
  assert.equal(applyCalibrationSnapshot({},null),false);
});

test("persisted calibration requires accepted finite complete models",()=>{
  const record={schema:4,featureMode:5,modelX:[1,2,3,4,5],modelY:[1,2,3,4,5],leftX:[1,2,3,4,5],leftY:[1,2,3,4,5],rightX:[1,2,3,4,5],rightY:[1,2,3,4,5],meta:{validation:{accepted:true,median:.04,p95:.1}}};
  assert.equal(calibrationRecordIsValid(record),true);
  assert.equal(calibrationRecordIsValid({...record,modelX:[1,2,NaN,4,5]}),false);
  assert.equal(calibrationRecordIsValid({...record,meta:{validation:{accepted:false,median:.04,p95:.1}}}),false);
  assert.equal(calibrationRecordIsValid({...record,meta:{validation:{accepted:true,median:.04,p95:.2}}}),false);
});

test("pupil analysis locates a synthetic dark eye region",()=>{
  const width=20,height=10,data=new Uint8ClampedArray(width*height*4).fill(255);
  for(let y=2;y<=7;y++)for(let x=3;x<=9;x++){const i=(y*width+x)*4;data[i]=data[i+1]=data[i+2]=18;data[i+3]=255}
  const pupil=analyzePupil({data,width,height});
  assert.ok(pupil.confidence>.2);
  assert.ok(Math.abs(pupil.x-6/19)<.05);
  assert.ok(Math.abs(pupil.y-4.5/9)<.06);
});

test("pupil tracker smooths plausible motion and rejects jumps",()=>{
  const previous={x:.4,y:.5,confidence:.8},current={x:.46,y:.52,confidence:.9};
  const tracked=trackPupil(previous,current);
  assert.equal(tracked.reason,"tracked");
  assert.ok(tracked.value.x>.4&&tracked.value.x<.46);
  assert.equal(trackPupil(previous,{x:.95,y:.05,confidence:.9}).reason,"jump");
  assert.equal(trackPupil(previous,{x:.42,y:.5,confidence:.1}).reason,"low-confidence");
});

test("anatomical pupil detector tolerates gradients, a glint and eyelid distractor",()=>{
  const width=40,height=20,data=new Uint8ClampedArray(width*height*4);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){const i=(y*width+x)*4,v=150+Math.round(x/width*70);data[i]=data[i+1]=data[i+2]=v;data[i+3]=255}
  for(let x=2;x<38;x++){const i=(2*width+x)*4;data[i]=data[i+1]=data[i+2]=15}
  for(let y=6;y<=14;y++)for(let x=12;x<=22;x++){if(((x-17)/5.5)**2+((y-10)/4.5)**2<=1){const i=(y*width+x)*4;data[i]=data[i+1]=data[i+2]=22}}
  {const i=(8*width+16)*4;data[i]=data[i+1]=data[i+2]=250}
  const pupil=analyzePupil({data,width,height});
  assert.ok(pupil&&pupil.confidence>.2);
  assert.ok(Math.abs(pupil.x-17/39)<.08);
  assert.ok(Math.abs(pupil.y-10/19)<.1);
});

test("hybrid participation requires coverage, confidence and stable sources per target",()=>{
  const sample=(targetIndex,source,confidence=.7)=>({targetIndex,observation:{leftEye:{source},rightEye:{source},hybridConfidence:confidence}});
  const good=[...Array(10)].flatMap((_,i)=>[sample(0,i<8?"pupil":"landmark"),sample(1,i<8?"pupil":"landmark")]);
  const report=hybridCoverageReport(good);
  assert.equal(report.passed,true);
  const fallbackHeavy=[...Array(10)].map((_,i)=>sample(0,i<4?"pupil":"landmark"));
  assert.equal(hybridCoverageReport(fallbackHeavy).passed,false);
  assert.equal(hybridCoverageReport([...Array(10)].map((_,i)=>sample(0,i%2?"pupil":"landmark"))).passed,false);
});

test("hybrid participation rejects persistent one-eye fallback",()=>{
  const samples=[...Array(20)].map(()=>({targetIndex:0,observation:{leftEye:{source:"pupil",pupilConfidence:.8},rightEye:{source:"landmark"},hybridConfidence:.8}}));
  const report=hybridCoverageReport(samples);
  assert.equal(report.left.coverage,1);
  assert.equal(report.right.coverage,0);
  assert.equal(report.binocularCoverage,0);
  assert.equal(report.passed,false);
});

test("patch translation recovers small image-space motion",()=>{
  const width=24,height=16,make=(ox,oy)=>{const data=new Uint8ClampedArray(width*height*4).fill(220);for(let y=5+oy;y<10+oy;y++)for(let x=8+ox;x<14+ox;x++){const i=(y*width+x)*4;data[i]=data[i+1]=data[i+2]=20;data[i+3]=255}return data};
  const flow=estimatePatchTranslation(make(0,0),make(2,1),width,height);
  assert.ok(Math.abs(flow.dx-2/(width-1))<1e-9);
  assert.ok(Math.abs(flow.dy-1/(height-1))<1e-9);
  assert.ok(flow.confidence>.1);
});

test("temporal pupil tracking rejects unstable component shape",()=>{
  const previous={x:.5,y:.5,confidence:.8,aspect:1},current={x:.51,y:.5,confidence:.8,aspect:3};
  assert.equal(trackPupil(previous,current).reason,"shape-instability");
});

test("shared luminance conversion downsamples deterministic grayscale",()=>{
  const rgba=new Uint8ClampedArray([255,0,0,255,0,255,0,255,0,0,255,255,255,255,255,255]);
  const gray=rgbaToLuminance(rgba,2,2);
  assert.deepEqual([...gray.data],[76,150,29,255]);
});

test("whole-frame luminance flow is bounded and detects translation",()=>{
  const width=20,height=16,make=(dx,dy)=>{const data=new Uint8ClampedArray(width*height).fill(220);for(let y=5+dy;y<10+dy;y++)for(let x=6+dx;x<12+dx;x++)data[y*width+x]=20;return data};
  const flow=estimateLuminanceTranslation(make(0,0),make(2,1),width,height,{maxShift:3,sampleStep:1});
  assert.deepEqual([flow.dx,flow.dy],[2,1]);assert.ok(flow.confidence>.1);
});

test("head keyframe bank is bounded and chooses the closest pose",()=>{
  let bank=[];for(let i=0;i<7;i++)bank=addHeadKeyframe(bank,{yaw:i*10,pitch:0,roll:0,x:0,y:0,z:5},{limit:5,minSeparation:0,now:1000+i});
  assert.equal(bank.length,5);const selected=selectHeadKeyframe(bank,{yaw:51,pitch:1,roll:0,x:0,y:0,z:5});assert.equal(selected.keyframe.head.yaw,50);assert.equal(selected.compatible,true);
});

test("head.raw parser accepts triangle rows and rejects malformed rows",()=>{
  const parsed=parseHeadRaw("0 0 0 100 0 0 0 100 0\n",{scale:.01});
  assert.equal(parsed.triangleCount,1);assert.equal(parsed.valid,true);assert.deepEqual(parsed.bounds,{min:[0,0,0],max:[1,1,0]});
  const malformed=parseHeadRaw("0 0 0 100 0 0 0 100 0\nnot a triangle\n",{scale:.01});assert.equal(malformed.valid,false);assert.equal(malformed.triangleCount,0);
});

test("face flow translates landmarks, decays and expires at the TTL",()=>{
  const marks=[{x:.4,y:.3},{x:.6,y:.5}],flow={dx:2,dy:-1,confidence:.8};
  const held=propagateLandmarks(marks,flow,100,50,100);assert.equal(held.source,"face-flow");assert.ok(Math.abs(held.landmarks[0].x-.42)<1e-9);assert.ok(Math.abs(held.landmarks[0].y-.28)<1e-9);assert.ok(Math.abs(held.confidence-.48)<1e-9);
  assert.equal(propagateLandmarks(marks,flow,100,50,251),null);
  assert.equal(propagateLandmarks(marks,{...flow,confidence:.05},100,50,20),null);
});

test("synthetic face-flow observations are never accepted",()=>{
  assert.equal(observationMayBeAccepted({source:"mediapipe",synthetic:false}),true);
  assert.equal(observationMayBeAccepted({source:"face-flow",synthetic:true}),false);
});

test("eye crops reuse the shared luminance frame and return RGBA",()=>{
  const frame={width:4,height:2,data:new Uint8ClampedArray([10,20,30,40,50,60,70,80])},patch=extractLuminancePatch(frame,{x:.25,y:0,w:.5,h:1},2,2);
  assert.deepEqual([...patch.data],[20,20,20,255,30,30,30,255,60,60,60,255,70,70,70,255]);
});

test("flow confidence rejects flat ambiguous frames",()=>{
  const a=new Uint8ClampedArray(20*16).fill(100),flow=estimateLuminanceTranslation(a,a,20,16,{maxShift:2});
  assert.ok(flow.confidence<.1);assert.ok(flow.texture<.01);assert.equal(flow.ambiguity,0);
});

test("restored keyframes are validated and carry descriptors",()=>{
  const head={yaw:0,pitch:0,roll:0,x:0,y:0,z:5},bank=addHeadKeyframe([],head,{eyes:{left:{x:.1,y:.2},right:{x:.3,y:.4}},source:"auto-stable",quality:.8,now:1});
  const valid=validateHeadKeyframes([...bank,{id:"bad",head:{yaw:NaN}}]);assert.equal(valid.length,1);assert.equal(valid[0].source,"auto-stable");assert.equal(valid[0].eyes.left.x,.1);
});

test("head geometry prior reports a bounded stabilization weight",()=>{
  const geometry=parseHeadRaw("0 0 0 100 0 0 0 100 0"),metric=headGeometryRegularizer(geometry,[{x:.2,y:.2},{x:.8,y:.8}]);
  assert.equal(metric.source,"user-calibration-landmarks");assert.ok(metric.weight>=0&&metric.weight<=.18);
});

test("symmetric 3x3 eigensolver returns descending orthonormal PCA axes",()=>{
  const e=symmetricEigen3([[9,0,0],[0,4,0],[0,0,1]]);assert.deepEqual(e.values,[9,4,1]);
  assert.ok(Math.abs(Math.hypot(e.vectors[0].x,e.vectors[0].y,e.vectors[0].z)-1)<1e-9);
  assert.ok(Math.abs(e.vectors[0].x*e.vectors[1].x+e.vectors[0].y*e.vectors[1].y+e.vectors[0].z*e.vectors[1].z)<1e-9);
});

test("pairwise nose scale and PCA frame are finite and sign-stable",()=>{
  const marks=Array.from({length:478},()=>({x:.5,y:.5,z:0}));PYTHON_NOSE_INDICES.forEach((id,i)=>marks[id]={x:.42+(i%6)*.025,y:.35+Math.floor(i/6)*.035,z:(i%3)*.008});marks[468]={x:.43,y:.45,z:-.02};marks[473]={x:.57,y:.45,z:-.02};
  const frame=nosePcaFrame(marks,640,480);assert.ok(frame&&frame.scale>0);assert.ok(meanPairwiseScale(frame.points)>0);
  const next=nosePcaFrame(marks,640,480,frame);assert.ok(next.axes.every((axis,i)=>axis.x*frame.axes[i].x+axis.y*frame.axes[i].y+axis.z*frame.axes[i].z>0));
});

test("Python-style eye sphere lock produces moving pseudo-3D slopes and atan2 angles",()=>{
  const marks=Array.from({length:478},()=>({x:.5,y:.5,z:0}));PYTHON_NOSE_INDICES.forEach((id,i)=>marks[id]={x:.42+(i%6)*.025,y:.35+Math.floor(i/6)*.035,z:(i%3)*.008});marks[468]={x:.43,y:.45,z:-.02};marks[473]={x:.57,y:.45,z:-.02};
  const frame=nosePcaFrame(marks,640,480),lock=lockEyeSpheres(marks,640,480,frame),center=updatePseudo3DGaze(marks,640,480,frame,lock,null,0);assert.ok(center&&center.direction.z<0);assert.ok(Math.abs(center.yaw)<1e-8&&Math.abs(center.pitch)<1e-8);
  const moved=structuredClone(marks);moved[468].x+=.02;moved[473].x+=.02;const result=updatePseudo3DGaze(moved,640,480,frame,lock,null,16),slope=pseudo3DRaySlope(result.direction);assert.ok(slope.x>0);assert.ok(result.yaw<0);
});

test("pseudo-3D display projection moves freely left right up and down",()=>{
  const start={x:200,y:120},span=34,center=pseudo3DDisplayTrace({x:0,y:0,z:-1},start,span),left=pseudo3DDisplayTrace({x:-.035,y:0,z:-.999},start,span),right=pseudo3DDisplayTrace({x:.035,y:0,z:-.999},start,span),up=pseudo3DDisplayTrace({x:0,y:-.035,z:-.999},start,span),down=pseudo3DDisplayTrace({x:0,y:.035,z:-.999},start,span);
  assert.deepEqual(center.end,start);
  assert.ok(right.end.x-left.end.x>35);
  assert.ok(up.end.y<start.y-50&&down.end.y>start.y+50);
  assert.ok(left.end.x<center.end.x&&right.end.x>center.end.x);
});

test("Python gaze ray uses perspective direction and collapses frontal gaze to a dot",()=>{
  const center={x:320,y:240,z:10},ray=projectPythonGazeRay(center,{x:.3,y:-.4,z:-.5},640,480,320,240,{gazeLength:130});
  assert.deepEqual(ray.start,{x:160,y:120});
  assert.ok(ray.end.x>ray.start.x&&ray.end.y<ray.start.y);
  assert.ok(Math.hypot(ray.end.x-ray.start.x,ray.end.y-ray.start.y)<=145.000001);
  const frontal=projectPythonGazeRay(center,{x:0,y:0,z:-1},640,480,320,240);
  assert.equal(frontal.neutral,true);assert.deepEqual(frontal.end,frontal.start);
  const up=projectPythonGazeRay(center,{x:0,y:-.2,z:-.98},640,480,640,480),down=projectPythonGazeRay(center,{x:0,y:.2,z:-.98},640,480,640,480);
  assert.ok(up.end.y<up.start.y&&down.end.y>down.start.y);
  const left=projectPythonGazeRay(center,{x:-.2,y:0,z:-.98},640,480,640,480),right=projectPythonGazeRay(center,{x:.2,y:0,z:-.98},640,480,640,480);
  assert.ok(left.end.x<left.start.x&&right.end.x>right.start.x);
});

test("pseudo-3D guards reject malformed frames, implausible scale and rear-facing rays",()=>{
  assert.equal(nosePcaFrame([],640,480),null);assert.equal(lockEyeSpheres([],640,480,null),null);assert.equal(pseudo3DRaySlope({x:0,y:0,z:.2}),null);
  const marks=Array.from({length:478},()=>({x:.5,y:.5,z:0}));PYTHON_NOSE_INDICES.forEach((id,i)=>marks[id]={x:.42+(i%6)*.025,y:.35+Math.floor(i/6)*.035,z:(i%3)*.008});marks[468]={x:.43,y:.45,z:-.02};marks[473]={x:.57,y:.45,z:-.02};const frame=nosePcaFrame(marks,640,480),lock=lockEyeSpheres(marks,640,480,frame);assert.equal(updatePseudo3DGaze(marks,640,480,{...frame,scale:frame.scale*2},lock,null,0),null);
});

test("signed binocular agreement rejects divergent and opposing rays",()=>{
  assert.ok(binocularRayConfidence({x:0,y:0,z:-1},{x:.05,y:0,z:-.999})>.9);
  assert.equal(binocularRayConfidence({x:.8,y:0,z:-.6},{x:-.8,y:0,z:-.6}),0);
  assert.equal(binocularRayConfidence({x:0,y:0,z:-1},{x:0,y:0,z:1}),0);
});

test("pseudo-3D confidence styling gates weak and stale directions",()=>{
  assert.equal(pseudo3DTraceVisualStyle({confidence:.34,at:100,now:100}),null);
  assert.equal(pseudo3DTraceVisualStyle({confidence:.9,at:100,now:400}),null);
  const weak=pseudo3DTraceVisualStyle({confidence:.4,at:100,now:120}),strong=pseudo3DTraceVisualStyle({confidence:.95,at:100,now:120});
  assert.ok(weak.opacity<strong.opacity);assert.ok(weak.dash.length>0);assert.equal(strong.dash.length,0);
});

test("eye-sphere lock invalidates after face loss and reacquisition discontinuities",()=>{
  const axes=[{x:1,y:0,z:0},{x:0,y:1,z:0},{x:0,y:0,z:1}],lock={calibrationScale:10,calibrationCenter:{x:0,y:0,z:0},calibrationAxes:axes};
  const frame={scale:10,center:{x:0,y:0,z:0},axes};
  assert.equal(eyeSphereLockInvalidationReason(lock,null,{faceLostMs:901}),"sustained-face-loss");
  assert.equal(eyeSphereLockInvalidationReason(lock,{...frame,scale:14}),"scale-discontinuity");
  assert.equal(eyeSphereLockInvalidationReason(lock,{...frame,center:{x:20,y:0,z:0}}),"position-discontinuity");
  assert.equal(eyeSphereLockInvalidationReason(lock,{...frame,axes:[{x:0,y:1,z:0},axes[1],axes[2]]}),"rotation-discontinuity");
  assert.equal(eyeSphereLockInvalidationReason(lock,frame),null);
});

test("near-degenerate nose PCA is rejected instead of flipping axes",()=>{
  const marks=Array.from({length:478},()=>({x:.5,y:.5,z:0}));
  PYTHON_NOSE_INDICES.forEach((id,i)=>{const a=i/PYTHON_NOSE_INDICES.length*Math.PI*2;marks[id]={x:.5+.1*Math.cos(a),y:.5+.1*Math.sin(a),z:0}});
  assert.equal(nosePcaFrame(marks,640,640),null);
});

test("time-based EMA has nearly equal one-second response at 15, 30 and 60 FPS",()=>{
  const response=fps=>{let remaining=1;for(let i=0;i<fps;i++)remaining*=1-exponentialSmoothingAlpha(1000/fps,95);return 1-remaining};
  const values=[15,30,60].map(response);assert.ok(Math.max(...values)-Math.min(...values)<1e-10);
});
test("calibration prelude binds the learned yaw sign without an undefined shorthand", () => {
  const app = fs.readFileSync(new URL("../dist/app.js", import.meta.url), "utf8");
  assert.match(app, /rightSign:yawSign/);
  assert.doesNotMatch(app, /axis:"yaw",rightSign,/);
});

test("weak screen-light response is classified and allowed to continue calibration", () => {
  const app = fs.readFileSync(new URL("../dist/app.js", import.meta.url), "utf8");
  assert.match(app, /status=!available\?"UNAVAILABLE"/);
  assert.match(app, /"PASS":"WEAK"/);
  assert.match(app, /Screen-light variation was weak; calibration continues/);
  assert.match(app, /error\.message\.startsWith\("Screen-light response was too weak"\)/);
  assert.match(app, /await captureTarget\(\);return/);
});

test("hybrid I-VT/I-DT classifier emits a fixation then a measured saccade", () => {
  const options={width:1000,height:500,minFixationMs:100,highVelocity:.65,lowVelocity:.36,settleSamples:2};
  let state=null,events=[];
  const feed=(x,y,at,confidence=.9)=>{const result=updateGazeEventClassifier(state,{x,y,at,confidence},options);state=result.state;events.push(...result.events)};
  feed(500,250,0);feed(502,250,50);feed(499,251,100);feed(501,249,150);
  feed(800,250,170);feed(900,250,190);feed(902,251,240);feed(903,251,290);
  assert.equal(events[0].type,"fixation");
  assert.ok(events[0].dwellMs>=100);
  const saccade=events.find(event=>event.type==="saccade");
  assert.ok(saccade);
  assert.ok(saccade.amplitudePx>300);
  assert.ok(saccade.peakVelocityNormalizedPerSec>.65);
  assert.ok(Math.abs(saccade.directionDeg)<5);
});

test("hybrid gaze classifier flushes an incomplete terminal saccade", () => {
  const options={width:1000,height:500,minFixationMs:100};
  let state=null;
  for(const sample of [{x:100,y:100,at:0,confidence:.9},{x:101,y:100,at:60,confidence:.9},{x:102,y:100,at:120,confidence:.9},{x:700,y:100,at:140,confidence:.9}])state=updateGazeEventClassifier(state,sample,options).state;
  const flushed=flushGazeEventClassifier(state,180,options);
  assert.equal(flushed.events[0].type,"saccade");
  assert.equal(flushed.events[0].incomplete,true);
});

test("saccade ends at the first settled sample and seeds fixation with the settle buffer", () => {
  const options={width:1000,height:500,minFixationMs:80,settleSamples:2};let state=null,events=[];
  const feed=(x,at)=>{const result=updateGazeEventClassifier(state,{x,y:250,at,confidence:.95},options);state=result.state;events.push(...result.events)};
  feed(100,0);feed(101,50);feed(100,100);feed(700,120);feed(800,140);feed(801,190);feed(802,240);
  const saccade=events.find(event=>event.type==="saccade");
  assert.equal(saccade.endedAt,190);
  assert.equal(state.fixation.started,190);
  assert.equal(state.fixation.count,2);
});

test("very-low-confidence spikes are excluded from gaze-event geometry", () => {
  const options={width:1000,height:500,minFixationMs:80};let state=null;
  for(const sample of [{x:100,y:100,at:0,confidence:.9},{x:101,y:100,at:50,confidence:.9},{x:900,y:400,at:70,confidence:.1}])state=updateGazeEventClassifier(state,sample,options).state;
  assert.equal(state.mode,"fixation");
  assert.equal(state.fixation.count,2);
  const first=updateGazeEventClassifier(null,{x:900,y:400,at:0,confidence:.1},options);
  assert.equal(first.state,null);
  assert.equal(first.rejectedLowConfidence,1);
});

test("delayed flush ends events at the last accepted sample", () => {
  const options={width:1000,height:500,minFixationMs:80};let state=null;
  for(const sample of [{x:100,y:100,at:0,confidence:.9},{x:101,y:100,at:60,confidence:.9},{x:102,y:100,at:120,confidence:.9},{x:700,y:100,at:140,confidence:.9}])state=updateGazeEventClassifier(state,sample,options).state;
  const event=flushGazeEventClassifier(state,5000,options).events[0];
  assert.equal(event.endedAt,140);
  assert.equal(event.durationMs,20);
  assert.equal(event.flushedAt,5000);
});

test("settle samples must be spatially coherent before closing a saccade", () => {
  const options={width:1000,height:500,minFixationMs:80,settleSamples:2};let state=null,events=[];
  const feed=(x,y,at)=>{const result=updateGazeEventClassifier(state,{x,y,at,confidence:.95},options);state=result.state;events.push(...result.events)};
  feed(100,100,0);feed(101,100,60);feed(100,100,120);feed(700,100,140);feed(701,100,190);feed(781,100,430);feed(782,100,480);
  const saccade=events.find(event=>event.type==="saccade");
  assert.ok(saccade);
  assert.equal(saccade.endedAt,430);
  assert.equal(state.fixation.started,430);
});

test("recording integration gates gaze events and freezes completed package", () => {
  const app=fs.readFileSync(new URL("../dist/app.js",import.meta.url),"utf8");
  assert.match(app,/function updateFixation\(x,y,confidence\)\{if\(state\.recording&&confidence<\.25\)/);
  assert.doesNotMatch(app,/function updateFixation\(x,y,confidence\)\{if\(!state\.recording\)return/);
  assert.match(app,/points=state\.livePathPoints\.map\(p=>\(\{\.\.\.p,x:p\.nx\*innerWidth,y:p\.ny\*innerHeight\}\)\)/);
  assert.match(app,/p\.provisional\?"LIVE"/);
  assert.match(app,/if\(!state\.recording&&!force\)continue/);
  assert.match(app,/if\(p\.breakBefore\)continue/);
  assert.match(app,/state\.livePathBreak=true/);
  assert.match(app,/recorded:state\.recording\|\|force/);
  assert.match(app,/else\{flushFixation\(true\);state\.livePathBreak=true;stopVideoRecording/);
  assert.match(app,/function sessionPackage\(\)\{const viewport=.*;return\{/);
  assert.doesNotMatch(app,/function sessionPackage\(\)\{flushFixation\(\)/);
});

test("live scan path is visible by default and redraws on resize", () => {
  const app=fs.readFileSync(new URL("../dist/app.js",import.meta.url),"utf8");
  const html=fs.readFileSync(new URL("../dist/index.html",import.meta.url),"utf8");
  assert.match(app,/pathEnabled:true/);
  assert.match(html,/id="pathToggle" type="checkbox" checked/);
  assert.match(html,/<canvas class="path-canvas" id="pathCanvas" aria-hidden="true"><\/canvas>/);
  assert.match(app,/addEventListener\("resize",\(\)=>\{sizeHeatmap\(\);sizePath\(\);if\(state\.pathEnabled\)drawGazePath\(\)/);
  assert.match(app,/if\(f&&now-f\.started>=120\)points\.push\(/);
  assert.match(app,/p\.provisional\?"LIVE":String\(i\+1\)/);
  assert.match(app,/livePathPoints\.push\(\{nx:event\.x,ny:event\.y/);
  assert.doesNotMatch(app,/nx:event\.x\/viewport\.width/);
  assert.match(app,/if\(gate\.holding\)\{flushFixation\(\);state\.livePathBreak=true;state\.fixation=null/);
  assert.match(app,/flushFixation\(\);state\.livePathBreak=true;state\.fixation=null/);
  assert.match(app,/if\(updated\.discontinuity\)state\.livePathBreak=true/);
});

test("classifier marks a long sample gap as a scan-path discontinuity", () => {
  const options={width:1000,height:500,minFixationMs:80};let state=null;
  for(const sample of [{x:100,y:100,at:0,confidence:.9},{x:101,y:100,at:80,confidence:.9}])state=updateGazeEventClassifier(state,sample,options).state;
  const resumed=updateGazeEventClassifier(state,{x:700,y:300,at:500,confidence:.9},options);
  assert.equal(resumed.discontinuity,true);
  assert.equal(resumed.state.fixation.x,700);
});

test("heatmap paints during live tracking and exports a visible composition", () => {
  const app=fs.readFileSync(new URL("../dist/app.js",import.meta.url),"utf8");
  const css=fs.readFileSync(new URL("../dist/extras.css",import.meta.url),"utf8");
  assert.match(app,/function heatmapPreviewThreshold\(\)\{return Math\.min\(heatmapDwellThreshold\(\),250\)\}/);
  assert.match(app,/if\(f&&now-f\.started>=previewMs\)drawHeatmapPreview/);
  assert.match(app,/if\(state\.heatmapEnabled&&event\.dwellMs>=threshold\)addHeatPoint/);
  assert.match(app,/function redrawHeatmap\(preview=null\)/);
  assert.match(app,/function drawHeatmapPreview\(x,y,confidence\)/);
  assert.match(app,/heatSamples:\[\]/);
  assert.match(app,/for\(const sample of state\.heatSamples\)paintHeatSample/);
  assert.match(app,/function compositedHeatmapCanvas\(\)/);
  assert.match(app,/ctx\.fillStyle="#f4f6f1"/);
  assert.match(app,/for\(const sample of state\.heatSamples\)paintHeatSample\(ctx,sample,innerWidth,innerHeight,dpr\)/);
  assert.match(app,/if\(!state\.heatPointCount\)return setMessage\("No heatmap density yet/);
  assert.match(app,/heatmap=state\.heatPointCount\?compositedHeatmapCanvas\(\)\.toDataURL\("image\/png"\):null/);
  assert.match(css,/mix-blend-mode:normal/);
});

test("adaptive drift RLS starts as a bounded offset and rejects unsafe residuals", () => {
  const initial=createAdaptiveDriftModel(),base={x:.4,y:.4},head={yaw:0,pitch:0,z:0};
  assert.deepEqual(predictAdaptiveDrift(initial,base,head).x,0);
  const first=updateAdaptiveDriftRls(initial,{base,target:{x:.46,y:.43},head,confidence:.9,at:1});
  assert.equal(first.accepted,true);
  assert.equal(first.model.mode,"offset");
  assert.ok(first.step<=.0200001);
  assert.ok(Math.hypot(predictAdaptiveDrift(first.model,base,head).x,predictAdaptiveDrift(first.model,base,head).y)<=.1200001);
  const rejected=updateAdaptiveDriftRls(first.model,{base,target:{x:.9,y:.9},head,confidence:.9,at:2});
  assert.equal(rejected.accepted,false);
  assert.equal(rejected.reason,"residual-too-large");
});

test("adaptive drift RLS unlocks pose-aware mode after diverse trusted anchors", () => {
  let model=createAdaptiveDriftModel();
  const anchors=[
    {base:{x:.3,y:.3},target:{x:.34,y:.31},head:{yaw:-4,pitch:0,z:0}},
    {base:{x:.7,y:.3},target:{x:.73,y:.32},head:{yaw:3,pitch:0,z:0}},
    {base:{x:.5,y:.7},target:{x:.54,y:.71},head:{yaw:8,pitch:3,z:.2}},
    {base:{x:.25,y:.72},target:{x:.28,y:.73},head:{yaw:-8,pitch:-6,z:-.4}},
    {base:{x:.76,y:.68},target:{x:.79,y:.69},head:{yaw:9,pitch:6,z:.4}},
    {base:{x:.28,y:.48},target:{x:.31,y:.49},head:{yaw:-7,pitch:5,z:-.3}},
    {base:{x:.72,y:.5},target:{x:.75,y:.51},head:{yaw:7,pitch:-5,z:.3}},
    {base:{x:.48,y:.24},target:{x:.51,y:.25},head:{yaw:6,pitch:7,z:.25}},
  ];
  for(const anchor of anchors){
    model=updateAdaptiveDriftRls(model,{...anchor,confidence:.9,at:model.count+1}).model;
    if(model.count===4){assert.equal(model.mode,"spatial-ready");assert.deepEqual(model.thetaX.slice(1,3),[0,0]);assert.deepEqual(model.thetaY.slice(1,3),[0,0])}
    if(model.count===5){assert.equal(model.mode,"spatial-rls");assert.ok(model.thetaX.slice(1,3).some(v=>v!==0)||model.thetaY.slice(1,3).some(v=>v!==0))}
    if(model.count===7){assert.equal(model.mode,"pose-ready");assert.deepEqual(model.thetaX.slice(3),[0,0,0]);assert.deepEqual(model.thetaY.slice(3),[0,0,0])}
  }
  assert.equal(model.mode,"pose-rls");
  assert.equal(model.accepted,8);
  assert.equal(validateAdaptiveDriftModel(model),true);
  assert.equal(adaptiveDriftFeatures({x:.5,y:.5},{yaw:26,pitch:22,z:3}).length,6);
  const correction=predictAdaptiveDrift(model,{x:.5,y:.5},{yaw:20,pitch:10,z:.5});
  assert.ok(Number.isFinite(correction.x)&&Number.isFinite(correction.y));
  assert.ok(Math.hypot(correction.x,correction.y)<=.1200001);
});

test("adaptive drift integration trains from stable baseline samples and supports exact rollback", () => {
  const app=fs.readFileSync(new URL("../dist/app.js",import.meta.url),"utf8");
  assert.match(app,/driftBuffer\.push\(\{at:driftNow,base:\{\.\.\.result\.baseline\}/);
  assert.match(app,/window\.length<10\|\|window\.at\(-1\)\.at-window\[0\]\.at<400/);
  assert.match(app,/s\.confidence>=\.75&&s\.binocular&&s\.direct&&s\.motion<=\.04/);
  assert.match(app,/beforeModel=structuredClone\(state\.adaptiveDriftModel\)/);
  assert.match(app,/state\.adaptiveDriftModel=structuredClone\(pending\.audit\.beforeModel\)/);
  assert.match(app,/cancelPendingDrift\("tracking stopped"\)/);
  assert.match(app,/cancelPendingDrift\("adaptive drift disabled"\)/);
  assert.match(app,/beforeError<\.01/);
  assert.match(app,/afterError<=beforeError\+\.001/);
  assert.match(app,/adaptiveDriftRevision:state\.adaptiveDriftModel\.revision/);
  assert.match(app,/cancelPendingDrift\("calibration started"\)/);
  assert.match(app,/prior\.type==="center-health-correction"/);
  assert.match(app,/prior\.type!=="adaptive-rls"/);
  assert.match(app,/type:"adaptive-rls"/);
  assert.match(app,/cancelPendingDrift\("center correction started"\)/);
  assert.match(app,/values\.push\(estimate\.baseRaw\)/);
  assert.match(app,/state\.driftHistory\.indexOf\(pending\.audit\)/);
  assert.match(app,/if\(state\.pendingDrift\)return setMessage/);
  assert.match(app,/if\(state\.pendingDrift\|\|!state\.driftArmed/);
  assert.match(app,/state\.adaptiveDriftModel=createAdaptiveDriftModel\(\)/);
  assert.match(app,/ui\.resetDrift\.addEventListener\("click",resetAdaptiveDrift\)/);
});

