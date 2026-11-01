let cvApi;
importScripts("./patch-geometry.js?v=1");

async function boot() {
  try {
    importScripts("./vendor/opencv.js");
    cvApi = self.cv;
    if (!cvApi?.Mat) {
      await new Promise(resolve => {
        const previous = cvApi.onRuntimeInitialized;
        cvApi.onRuntimeInitialized = () => { previous?.(); resolve(); };
      });
    }
    if (!cvApi?.warpAffine || !cvApi?.cvtColor) throw new Error(`Required OpenCV.js functions are unavailable (warpAffine: ${typeof cvApi?.warpAffine}, cvtColor: ${typeof cvApi?.cvtColor}).`);
    console.log("OpenCV.js worker ready");
    postMessage({ type: "ready" });
  } catch (error) { postMessage({ type: "error", error: error.message || String(error) }); }
}

function patch(frame, points, width=36, height=20) {
  const geometry=PatchGeometry.canonicalPatchGeometry(points,width,height);
  if(!geometry)return null;
  const pointsForTransform=geometry.source,dst=geometry.destination;
  let srcPts, dstPts, transform, output, gray;
  try {
    srcPts=cvApi.matFromArray(3,1,cvApi.CV_32FC2,pointsForTransform); dstPts=cvApi.matFromArray(3,1,cvApi.CV_32FC2,dst);
    const solve = values => {
      const rows = [[pointsForTransform[0],pointsForTransform[1],1,values[0]],[pointsForTransform[2],pointsForTransform[3],1,values[1]],[pointsForTransform[4],pointsForTransform[5],1,values[2]]];
      for(let column=0;column<3;column++){
        let pivot=column; for(let row=column+1;row<3;row++)if(Math.abs(rows[row][column])>Math.abs(rows[pivot][column]))pivot=row;
        [rows[column],rows[pivot]]=[rows[pivot],rows[column]];
        const divisor=rows[column][column]; for(let index=column;index<4;index++)rows[column][index]/=divisor;
        for(let row=0;row<3;row++)if(row!==column){const factor=rows[row][column];for(let index=column;index<4;index++)rows[row][index]-=factor*rows[column][index];}
      }
      return rows.map(row=>row[3]);
    };
    transform=cvApi.matFromArray(2,3,cvApi.CV_64F,[...solve([dst[0],dst[2],dst[4]]),...solve([dst[1],dst[3],dst[5]])]); output=new cvApi.Mat();
    cvApi.warpAffine(frame,output,transform,new cvApi.Size(width,height),cvApi.INTER_LINEAR,cvApi.BORDER_REFLECT_101??cvApi.BORDER_REPLICATE,new cvApi.Scalar(0,0,0,255));
    gray=new cvApi.Mat(); cvApi.cvtColor(output,gray,cvApi.COLOR_RGBA2GRAY);
    const values=Array.from(gray.data,value=>value/255), average=values.reduce((sum,value)=>sum+value,0)/values.length;
    const deviation=Math.sqrt(values.reduce((sum,value)=>sum+(value-average)**2,0)/values.length)||1;
    return values.map(value=>(value-average)/deviation);
  } finally { srcPts?.delete(); dstPts?.delete(); transform?.delete(); output?.delete(); gray?.delete(); }
}

self.onmessage = event => {
  if (event.data?.type !== "patches" || !cvApi) return;
    const { id, width, height, pixels, left, right, face } = event.data; let frame;
  try {
    frame=cvApi.matFromArray(height,width,cvApi.CV_8UC4,new Uint8Array(pixels));
    postMessage({type:"patches",id,patches:[patch(frame,left),patch(frame,right),face?patch(frame,face,20,24):null]});
  } catch (error) { postMessage({type:"patches",id,error:error.message||String(error)}); }
  finally { frame?.delete(); }
};

boot();
