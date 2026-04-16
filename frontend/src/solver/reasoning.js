/**
 * Explainable AI (XAI) Reasoning Agent
 * Generates step-by-step instructions based on Graph Diffusion Solver results.
 */
export class ReasoningAgent {
  generateInstruction(transformation) {
    if (!transformation) return null;
    
    const { tx, ty, r } = transformation;
    
    let vertical = ty < 0 ? 'up' : 'down';
    let horizontal = tx < 0 ? 'left' : 'right';
    
    // Trivial distance check
    if (Math.abs(tx) < 10 && Math.abs(ty) < 10 && Math.abs(r) < 10) {
      return "Piece is almost correctly placed. Try to snap it into position.";
    }

    const rotationStr = Math.abs(r) > 15 
      ? `Rotate the piece ${Math.abs(Math.round(r))} degrees ${r > 0 ? 'clockwise' : 'counter-clockwise'} and ` 
      : '';

    return `${rotationStr}move it ${Math.round(Math.abs(tx))} units ${horizontal} and ${Math.round(Math.abs(ty))} units ${vertical}.`;
  }
}
