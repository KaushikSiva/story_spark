import { useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, BookOpen, Star, Home, RotateCcw, Play, Pause } from "lucide-react";

interface Scene {
  filename: string;
  image_url: string;
  index: number;
  text: string;
}

interface StorybookData {
  count: number;
  scenes: Scene[];
  style: string;
}

interface StorybookSlideshowProps {
  storybook: StorybookData;
  onBack: () => void;
}

export function StorybookSlideshow({ storybook, onBack }: StorybookSlideshowProps) {
  const [currentSceneIndex, setCurrentSceneIndex] = useState(0);
  const [imageLoading, setImageLoading] = useState(true);
  const [isAutoplay, setIsAutoplay] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const [frameHeight, setFrameHeight] = useState<number | null>(null);

  useEffect(() => {
    const onFsChange = () => {
      if (!document.fullscreenElement) {
        setIsAutoplay(false);
      }
    };
    document.addEventListener("fullscreenchange", onFsChange);
    return () => document.removeEventListener("fullscreenchange", onFsChange);
  }, []);

  useEffect(() => {
    if (!isAutoplay) return;
    const id = window.setInterval(() => {
      setCurrentSceneIndex((prev) => {
        const lastIndex = storybook.scenes.length - 1;
        if (prev >= lastIndex) {
          // Stop on last slide (no looping)
          setIsAutoplay(false);
          return prev;
        }
        setImageLoading(true);
        return prev + 1;
      });
    }, 4000);
    return () => clearInterval(id);
  }, [isAutoplay, storybook.scenes.length]);

  // Dynamically size the image frame to the remaining viewport space in autoplay/fullscreen
  useEffect(() => {
    const calc = () => {
      if (!isAutoplay) {
        setFrameHeight(null);
        return;
      }
      const el = frameRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const vh = window.innerHeight || document.documentElement.clientHeight;
      const remaining = Math.max(200, Math.floor(vh - rect.top - 12)); // small bottom margin
      setFrameHeight(remaining);
    };
    calc();
    window.addEventListener('resize', calc);
    document.addEventListener('fullscreenchange', calc);
    return () => {
      window.removeEventListener('resize', calc);
      document.removeEventListener('fullscreenchange', calc);
    };
  }, [isAutoplay]);

  const currentScene = storybook.scenes[currentSceneIndex];
  const isFirstScene = currentSceneIndex === 0;
  const isLastScene = currentSceneIndex === storybook.scenes.length - 1;

  const goToNextScene = () => {
    if (!isLastScene) {
      setCurrentSceneIndex(prev => prev + 1);
      setImageLoading(true);
    }
  };

  const goToPreviousScene = () => {
    if (!isFirstScene) {
      setCurrentSceneIndex(prev => prev - 1);
      setImageLoading(true);
    }
  };

  const goToScene = (index: number) => {
    setCurrentSceneIndex(index);
    setImageLoading(true);
  };

  const resetStory = () => {
    setCurrentSceneIndex(0);
    setImageLoading(true);
  };

  const startAutoplay = async () => {
    setIsAutoplay(true);
    const el = containerRef.current;
    if (el && !document.fullscreenElement && el.requestFullscreen) {
      try {
        await el.requestFullscreen();
      } catch {}
    }
  };

  const stopAutoplay = async () => {
    setIsAutoplay(false);
    if (document.fullscreenElement && document.exitFullscreen) {
      try {
        await document.exitFullscreen();
      } catch {}
    }
  };

  return (
    <div ref={containerRef} className="min-h-screen bg-gradient-to-br from-yellow-50 via-orange-50 to-red-50">
      {/* Header */}
      <div className="sticky top-0 bg-white/90 backdrop-blur-lg border-b-4 border-orange-200 z-40 shadow-lg">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <button
              onClick={onBack}
              className="inline-flex items-center px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white font-bold rounded-full shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200"
            >
              <Home className="h-5 w-5 mr-2" />
              Home
            </button>
            
            <div className="text-center">
              <h1 className="text-2xl md:text-3xl font-bold text-gray-800 flex items-center justify-center">
                <BookOpen className="h-8 w-8 text-green-500 mr-3" />
                Story Time!
              </h1>
              <p className="text-gray-600 font-medium">Scene {currentSceneIndex + 1} of {storybook.scenes.length}</p>
            </div>
            
            <div className="flex items-center gap-2">
              {isAutoplay ? (
                <button
                  onClick={stopAutoplay}
                  className="inline-flex items-center px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white font-bold rounded-full shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200"
                >
                  <Pause className="h-5 w-5 mr-2" />
                  Stop
                </button>
              ) : (
                <button
                  onClick={startAutoplay}
                  className="inline-flex items-center px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white font-bold rounded-full shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200"
                >
                  <Play className="h-5 w-5 mr-2" />
                  Autoplay
                </button>
              )}
              <button
                onClick={resetStory}
                className="inline-flex items-center px-4 py-2 bg-green-500 hover:bg-green-600 text-white font-bold rounded-full shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200"
              >
                <RotateCcw className="h-5 w-5 mr-2" />
                Restart
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white rounded-3xl shadow-2xl overflow-hidden border-4 border-orange-200">
          {/* Progress Bar */}
          <div className="bg-gradient-to-r from-yellow-200 to-orange-200 p-4">
            <div className="flex items-center justify-center space-x-2">
              {storybook.scenes.map((_, index) => (
                <button
                  key={index}
                  onClick={() => goToScene(index)}
                  className={`w-4 h-4 rounded-full transition-all duration-300 transform hover:scale-125 ${
                    index === currentSceneIndex
                      ? 'bg-orange-500 scale-125 shadow-lg'
                      : index < currentSceneIndex
                      ? 'bg-green-400'
                      : 'bg-gray-300'
                  }`}
                />
              ))}
            </div>
          </div>

          {/* Scene Content */}
          <div className="p-6 md:p-8">
            {/* Image */}
            <div className="relative mb-6">
              <div
                ref={frameRef}
                className={`${isAutoplay ? "" : "aspect-square"} relative rounded-2xl overflow-hidden shadow-xl border-4 border-yellow-200`}
                style={isAutoplay && frameHeight ? { height: `${frameHeight}px` } : undefined}
              >
                {imageLoading && (
                  <div className="absolute inset-0 bg-gradient-to-br from-yellow-100 to-orange-100 flex items-center justify-center">
                    <div className="text-center space-y-4">
                      <div className="w-16 h-16 border-4 border-orange-400 border-t-transparent rounded-full animate-spin mx-auto"></div>
                      <p className="text-orange-600 font-bold text-lg">Loading magical scene...</p>
                    </div>
                  </div>
                )}
                <img
                  src={currentScene.image_url}
                  alt={`Scene ${currentScene.index}`}
                  className={`w-full h-full ${isAutoplay ? 'object-contain' : 'object-cover'} transition-opacity duration-300 ${
                    imageLoading ? 'opacity-0' : 'opacity-100'
                  }`}
                  onLoad={() => setImageLoading(false)}
                  onError={() => setImageLoading(false)}
                />
                {isAutoplay && (
                  <div className="absolute bottom-0 left-0 right-0 bg-black/50 text-white p-6 md:p-8">
                    <p className="text-lg md:text-2xl leading-relaxed text-center">
                      {currentScene.text}
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Story Text */}
            {!isAutoplay && (
              <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-2xl p-6 border-4 border-blue-200 shadow-inner">
                <p className="text-xl md:text-2xl text-gray-800 leading-relaxed font-medium text-center">
                  {currentScene.text}
                </p>
              </div>
            )}

            {/* Navigation */}
            {!isAutoplay && (
            <div className="flex items-center justify-between mt-8">
              <button
                onClick={goToPreviousScene}
                disabled={isFirstScene}
                className={`flex items-center px-6 py-4 rounded-full font-bold text-lg shadow-lg transform transition-all duration-200 ${
                  isFirstScene
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-purple-500 hover:bg-purple-600 text-white hover:shadow-xl hover:scale-105'
                }`}
              >
                <ChevronLeft className="h-6 w-6 mr-2" />
                Previous
              </button>

              <div className="flex items-center space-x-2">
                {Array.from({ length: 3 }, (_, i) => (
                  <Star
                    key={i}
                    className="h-8 w-8 text-yellow-400 animate-pulse"
                    style={{ animationDelay: `${i * 0.2}s` }}
                  />
                ))}
              </div>

              <button
                onClick={goToNextScene}
                disabled={isLastScene}
                className={`flex items-center px-6 py-4 rounded-full font-bold text-lg shadow-lg transform transition-all duration-200 ${
                  isLastScene
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-purple-500 hover:bg-purple-600 text-white hover:shadow-xl hover:scale-105'
                }`}
              >
                Next
                <ChevronRight className="h-6 w-6 ml-2" />
              </button>
            </div>
            )}
          </div>
        </div>

        {/* Story Complete Message */}
        {isLastScene && (
          <div className="mt-8 text-center">
            <div className="bg-gradient-to-r from-green-400 to-blue-500 text-white rounded-2xl p-6 shadow-2xl border-4 border-green-300">
              <h3 className="text-2xl font-bold mb-2">🎉 Story Complete! 🎉</h3>
              <p className="text-lg mb-4">Great job reading the whole story!</p>
              <button
                onClick={resetStory}
                className="inline-flex items-center px-6 py-3 bg-white text-green-600 font-bold rounded-full shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200"
              >
                <RotateCcw className="h-5 w-5 mr-2" />
                Read Again
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
