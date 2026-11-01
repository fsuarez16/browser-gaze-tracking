// Browser-native PCA math module. 
const average = values => values.reduce((sum, value) => sum + value, 0) / (values.length || 1);

export function fit(rows, componentCount = 12) {
  if (!Array.isArray(rows) || rows.length < 2 || !rows[0]?.length) throw new Error("PCA requires at least two numeric rows");
  const dimensions = rows[0].length;
  if (!rows.every(row => row.length === dimensions && row.every(Number.isFinite))) throw new Error("PCA rows must have equal numeric dimensions");
  const mean = Array.from({length: dimensions}, (_, column) => average(rows.map(row => row[column])));
  const centered = rows.map(row => row.map((value, column) => value - mean[column]));
  // Dual PCA: diagonalize the sample Gram matrix, then map eigenvectors back
  // to pixel space. This keeps 36x20x2 eye patches practical in the browser.
  const sampleCount=rows.length,gram=Array.from({length:sampleCount},(_,i)=>Array.from({length:sampleCount},(_,j)=>centered[i].reduce((sum,value,k)=>sum+value*centered[j][k],0)/sampleCount));
  const sampleVectors=[],components=[],eigenvalues=[];
  const limit = Math.min(componentCount, dimensions, sampleCount - 1);
  for (let componentIndex = 0; componentIndex < limit; componentIndex++) {
    let vector = Array.from({length: sampleCount}, (_, i) => Math.sin((i + 1) * (componentIndex + 1) * 1.61803398875) + 0.01);
    let norm = Math.hypot(...vector); vector = vector.map(value => value / norm);
    for (let iteration = 0; iteration < 100; iteration++) {
      let next = gram.map(row => row.reduce((sum, value, j) => sum + value * vector[j], 0));
      for (const previous of sampleVectors) {
        const dot = next.reduce((sum, value, j) => sum + value * previous[j], 0);
        next = next.map((value, j) => value - dot * previous[j]);
      }
      norm = Math.hypot(...next); if (norm < 1e-12) break;
      next = next.map(value => value / norm);
      const sign = next.reduce((sum, value, i) => sum + value * vector[i], 0) < 0 ? -1 : 1;
      next = next.map(value => value * sign);
      const delta = Math.hypot(...next.map((value, i) => value - vector[i]));
      vector = next; if (delta < 1e-9) break;
    }
    const gramVector = gram.map(row => row.reduce((sum, value, j) => sum + value * vector[j], 0));
    const eigenvalue = Math.max(0, vector.reduce((sum, value, j) => sum + value * gramVector[j], 0));
    if (eigenvalue < 1e-10) break;
    let component=Array.from({length:dimensions},(_,column)=>centered.reduce((sum,row,i)=>sum+row[column]*vector[i],0)),componentNorm=Math.hypot(...component);
    if(componentNorm<1e-12)break;component=component.map(value=>value/componentNorm);
    sampleVectors.push(vector);components.push(component); eigenvalues.push(eigenvalue);
  }
  if (!components.length) throw new Error("PCA could not find a varying eye component");
  return {mean, components, eigenvalues, backend: "Native JavaScript PCA math"};
}

export function project(row, model) {
  return model.components.map(component => component.reduce((sum, value, j) => sum + value * (row[j] - model.mean[j]), 0));
}

export function fuse(models, componentCount=12) {
  const valid=models.filter(model=>model?.mean?.length&&model.components?.length&&model.eigenvalues?.length);
  if(valid.length<2)return valid[0];
  const rows=[];for(const model of valid){rows.push([...model.mean]);model.components.forEach((component,index)=>{const scale=Math.sqrt(Math.max(0,model.eigenvalues[index]||0));rows.push(model.mean.map((value,j)=>value+scale*component[j]));rows.push(model.mean.map((value,j)=>value-scale*component[j]))})}
  const result=fit(rows,componentCount);result.backend="Native JavaScript fused PCA math";return result;
}

// Conditional reconstruction for a joint [input | scaled output] PCA model.
// It solves the latent coefficients from the observed input coordinates and
// reconstructs the two held-out output coordinates.
export function predictJoint(input, model) {
  const inputLength=model.inputLength,scale=model.outputScale||1;
  if(input.length!==inputLength||model.mean.length!==inputLength+2)throw new Error("Invalid joint PCA dimensions");
  const nm=model.normalizationMean||Array(model.mean.length).fill(0),ns=model.normalizationStd||Array(model.mean.length).fill(1),observed=input.map((v,i)=>Math.max(-5,Math.min(5,(v-nm[i])/(ns[i]||1))));
  const k=model.components.length,A=Array.from({length:k},(_,i)=>Array.from({length:k},(_,j)=>model.components[i].slice(0,inputLength).reduce((s,v,n)=>s+v*model.components[j][n],0)+(i===j?1e-6:0))),b=Array.from({length:k},(_,i)=>model.components[i].slice(0,inputLength).reduce((s,v,n)=>s+v*(observed[n]-model.mean[n]),0));
  for(let i=0;i<k;i++){let p=i;for(let r=i+1;r<k;r++)if(Math.abs(A[r][i])>Math.abs(A[p][i]))p=r;[A[i],A[p]]=[A[p],A[i]];[b[i],b[p]]=[b[p],b[i]];const d=A[i][i]||1e-9;for(let j=i;j<k;j++)A[i][j]/=d;b[i]/=d;for(let r=0;r<k;r++)if(r!==i){const q=A[r][i];for(let j=i;j<k;j++)A[r][j]-=q*A[i][j];b[r]-=q*b[i]}}
  const out=[0,1].map(o=>{const index=inputLength+o,reconstructed=model.mean[index]+model.components.reduce((s,c,i)=>s+b[i]*c[index],0);return(reconstructed*ns[index]+nm[index])/scale});
  return{x:out[0],y:out[1]};
}
