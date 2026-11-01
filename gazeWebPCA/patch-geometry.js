(function(root){
  function canonicalPatchGeometry(points,width=36,height=20){
    if(!Array.isArray(points)||points.length!==6||!points.every(Number.isFinite))return null;
    let ax=points[0],ay=points[1],bx=points[2],by=points[3];
    const hint={x:points[4],y:points[5]};
    if(ax>bx){[ax,bx]=[bx,ax];[ay,by]=[by,ay]}
    const dx=bx-ax,dy=by-ay,span=Math.hypot(dx,dy);
    if(span<4)return null;
    const midpoint={x:(ax+bx)/2,y:(ay+by)/2};
    let nx=-dy/span,ny=dx/span;
    if((hint.x-midpoint.x)*nx+(hint.y-midpoint.y)*ny<0){nx=-nx;ny=-ny}
    const source=[ax,ay,bx,by,midpoint.x+nx*span,midpoint.y+ny*span];
    const eye=width===36&&height===20;
    const destination=eye?[5,10,31,10,18,36]:[2,10,width-2,10,width/2,10-(width-4)];
    return{source,destination,span,kind:eye?"eye":"face"};
  }
  root.PatchGeometry=Object.freeze({canonicalPatchGeometry});
})(typeof self!=="undefined"?self:globalThis);
