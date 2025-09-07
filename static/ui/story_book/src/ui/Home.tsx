import { useState } from "react";
import { BookOpen, Star, Sparkles, Heart, Smile, Rainbow, Sun } from "lucide-react";
import { StorybookSlideshow } from "~/components/StorybookSlideshow";
import type { StorybookData } from "./storybookData";

export function Home() {
  const [showStorybook, setShowStorybook] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [story, setStory] = useState<StorybookData | null>(null);

  const handleStartStorybook = async () => {
    setError(null);
    setIsLoading(true);
    try {
      const res = await fetch("/api/storyboard");
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      const data = await res.json();
      const payload: StorybookData = {
        count: data.count,
        scenes: data.scenes,
        style: data.style ?? "",
      };
      setStory(payload);
      setShowStorybook(true);
    } catch (e: any) {
      setError(e?.message || "Failed to load storybook");
    } finally {
      setIsLoading(false);
    }
  };

  const handleBack = () => {
    setShowStorybook(false);
  };

  if (showStorybook && story) {
    return <StorybookSlideshow storybook={story} onBack={handleBack} />;
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-yellow-50 via-orange-50 to-red-50">
        <div className="text-center space-y-8">
          <div className="relative">
            <div className="w-32 h-32 border-8 border-orange-200 border-t-orange-500 rounded-full animate-spin mx-auto"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <BookOpen className="h-16 w-16 text-orange-500 animate-pulse" />
            </div>
          </div>
          <div className="space-y-4">
            <h2 className="text-4xl font-bold text-gray-800">Loading Your Story...</h2>
            <p className="text-xl text-gray-600">Getting ready for story time! 📚✨</p>
            <div className="flex justify-center space-x-2 mt-4">
              {[0, 1, 2].map((i) => (
                <div key={i} className="w-4 h-4 bg-orange-500 rounded-full animate-bounce" style={{ animationDelay: `${i * 0.2}s` }}></div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-yellow-50 via-orange-50 to-red-50">
        <div className="text-center space-y-6 max-w-md mx-auto px-4">
          <div className="text-6xl">😢</div>
          <h2 className="text-3xl font-bold text-gray-800">Failed to load storybook</h2>
          <p className="text-lg text-gray-600">{error}</p>
          <button
            onClick={handleStartStorybook}
            className="px-8 py-4 bg-orange-500 hover:bg-orange-600 text-white font-bold text-lg rounded-full shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200"
          >
            Try Again 🔄
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-yellow-100 via-orange-50 to-red-100"></div>
        <div className="absolute top-0 left-0 w-full h-full">
          <div className="absolute top-20 left-20 w-72 h-72 bg-blue-300 rounded-full mix-blend-multiply filter blur-xl opacity-60 animate-blob"></div>
          <div className="absolute top-40 right-20 w-72 h-72 bg-green-300 rounded-full mix-blend-multiply filter blur-xl opacity-60 animate-blob animation-delay-2000"></div>
          <div className="absolute -bottom-8 left-1/2 w-72 h-72 bg-purple-300 rounded-full mix-blend-multiply filter blur-xl opacity-60 animate-blob animation-delay-4000"></div>
        </div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center space-y-12">
            <div className="space-y-6">
              <div className="inline-flex items-center px-6 py-3 bg-white/80 backdrop-blur-sm rounded-full shadow-lg border-4 border-orange-200">
                <BookOpen className="h-6 w-6 text-orange-600 mr-3 animate-pulse" />
                <span className="text-orange-700 font-bold text-lg">Story Time Adventure</span>
              </div>
              <h1 className="text-5xl md:text-7xl font-bold text-gray-900 leading-tight">
                Kid's
                <span className="bg-gradient-to-r from-blue-600 via-green-500 to-purple-600 bg-clip-text text-transparent animate-gradient"> Storybook</span>
              </h1>
              <p className="text-xl md:text-2xl text-gray-600 max-w-4xl mx-auto leading-relaxed">
                Get ready for an amazing adventure! Click the button below to start reading a magical story with beautiful pictures and fun characters! 📚✨
              </p>
            </div>
            <div className="pt-8">
              <button
                onClick={handleStartStorybook}
                className="group relative px-12 py-6 bg-gradient-to-r from-orange-500 via-red-500 to-purple-500 hover:from-orange-600 hover:via-red-600 hover:to-purple-600 text-white font-bold text-2xl rounded-3xl shadow-2xl hover:shadow-orange-500/25 transform hover:scale-105 transition-all duration-300 flex items-center space-x-4 animate-pulse-glow mx-auto"
              >
                <BookOpen className="h-8 w-8" />
                <span>Start Reading! 📖</span>
                <Sparkles className="h-8 w-8 group-hover:rotate-12 transition-transform duration-300" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Features */}
      <div className="py-24 bg-white/70 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center space-y-12">
            <div className="space-y-4">
              <h2 className="text-4xl font-bold text-gray-900">Why Kids Love Our Stories! 🌟</h2>
              <p className="text-xl text-gray-600 max-w-3xl mx-auto">Every story is full of magic, adventure, and fun characters that kids absolutely love!</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {[
                { icon: Rainbow, title: "Colorful Pictures", description: "Beautiful, bright images that bring every story to life with amazing colors!", color: "from-red-400 to-pink-400", emoji: "🌈" },
                { icon: Heart, title: "Fun Characters", description: "Meet friendly robots, wise owls, and other amazing characters in every story!", color: "from-green-400 to-blue-400", emoji: "💝" },
                { icon: Star, title: "Easy Reading", description: "Simple words and exciting adventures perfect for young readers to enjoy!", color: "from-purple-400 to-indigo-400", emoji: "⭐" },
              ].map((feature, index) => (
                <div key={index} className="group">
                  <div className="bg-white/90 backdrop-blur-sm rounded-3xl p-8 shadow-xl border-4 border-yellow-200 hover:shadow-2xl transform hover:-translate-y-4 transition-all duration-300 card-hover">
                    <div className="text-center space-y-4">
                      <div className="text-6xl">{feature.emoji}</div>
                      <div className={`w-16 h-16 bg-gradient-to-br ${feature.color} rounded-full flex items-center justify-center mx-auto group-hover:scale-110 transition-transform duration-300`}>
                        <feature.icon className="h-8 w-8 text-white" />
                      </div>
                      <h3 className="text-2xl font-bold text-gray-900">{feature.title}</h3>
                      <p className="text-gray-600 leading-relaxed text-lg">{feature.description}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* How It Works */}
      <div className="py-24 bg-gradient-to-br from-yellow-50 via-orange-50 to-red-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center space-y-12">
            <h2 className="text-4xl font-bold text-gray-900">How Story Time Works! 📖</h2>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
              {[
                { step: "1", icon: Sun, title: "Click Start", description: "Press the big colorful button to begin!", emoji: "👆" },
                { step: "2", icon: Sparkles, title: "Watch Magic", description: "See beautiful pictures appear like magic!", emoji: "✨" },
                { step: "3", icon: BookOpen, title: "Read Together", description: "Enjoy the story with amazing pictures!", emoji: "👨‍👩‍👧‍👦" },
                { step: "4", icon: Smile, title: "Have Fun", description: "Laugh, learn, and enjoy the adventure!", emoji: "🎉" },
              ].map((step, index) => (
                <div key={index} className="text-center">
                  <div className="relative mb-6">
                    <div className="w-24 h-24 bg-gradient-to-br from-orange-400 to-red-400 rounded-full flex items-center justify-center mx-auto shadow-xl">
                      <step.icon className="h-12 w-12 text-white" />
                    </div>
                    <div className="absolute -top-2 -right-2 w-10 h-10 bg-blue-500 text-white rounded-full flex items-center justify-center text-lg font-bold border-4 border-white">{step.step}</div>
                  </div>
                  <div className="text-4xl mb-3">{step.emoji}</div>
                  <h3 className="text-xl font-bold text-gray-900 mb-3">{step.title}</h3>
                  <p className="text-gray-600 text-lg">{step.description}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* CTA */}
      <div className="py-24 bg-gradient-to-br from-purple-600 via-blue-600 to-green-600 text-white">
        <div className="max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
          <div className="space-y-8">
            <div className="text-6xl">🚀</div>
            <h2 className="text-4xl font-bold">Ready for Story Time Adventure?</h2>
            <p className="text-xl text-blue-100">Let's start reading and have lots of fun together! 🌟</p>
            <button
              onClick={handleStartStorybook}
              className="group inline-flex items-center px-12 py-6 bg-white text-purple-600 hover:text-purple-700 font-bold text-2xl rounded-3xl shadow-2xl hover:shadow-white/25 transform hover:scale-105 transition-all duration-300 space-x-4"
            >
              <BookOpen className="h-8 w-8" />
              <span>Let's Read Together!</span>
              <Sparkles className="h-8 w-8 group-hover:rotate-12 transition-transform duration-300" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
  
