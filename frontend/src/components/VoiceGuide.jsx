import React, { useEffect, useState, useCallback } from 'react';
import { Volume2, VolumeX } from 'lucide-react';

export const VoiceGuide = ({ instruction }) => {
  const [isMuted, setIsMuted] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  const speak = useCallback((text) => {
    if (isMuted || !text) return;
    
    window.speechSynthesis.cancel(); // Stop current speech
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.onstart = () => setIsPlaying(true);
    utterance.onend = () => setIsPlaying(false);
    window.speechSynthesis.speak(utterance);
  }, [isMuted]);

  useEffect(() => {
    if (instruction) speak(instruction);
  }, [instruction, speak]);

  return (
    <div className="flex items-center gap-3 bg-industrial-800 p-3 rounded-lg border border-industrial-700 shadow-md">
      <button 
        onClick={() => setIsMuted(!isMuted)}
        className={`p-2 rounded-full transition-colors ${isMuted ? 'bg-industrial-700 text-gray-500' : 'bg-industrial-700 text-accent hover:bg-industrial-900'}`}
      >
        {isMuted ? <VolumeX size={20} /> : <Volume2 size={20} />}
      </button>
      <div className="flex-1 text-sm font-medium p-1">
        {isPlaying ? (
          <span className="text-accent animate-pulse">{instruction}</span>
        ) : (
          <span className="text-gray-400">{instruction || "Waiting for inference..."}</span>
        )}
      </div>
    </div>
  );
};
