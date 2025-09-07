export interface Scene {
  filename: string;
  image_url: string;
  index: number;
  text: string;
}

export interface StorybookData {
  count: number;
  scenes: Scene[];
  style: string;
}

export const demoStorybook: StorybookData = {
  count: 4,
  scenes: [
    {
      filename: "scene_001.png",
      image_url:
        "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=800&h=600&fit=crop",
      index: 1,
      text:
        "Once upon a time, in a magical forest filled with colorful flowers and friendly animals, there lived a brave little rabbit named Luna.",
    },
    {
      filename: "scene_002.png",
      image_url:
        "https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=800&h=600&fit=crop",
      index: 2,
      text:
        "Luna loved to explore and help her forest friends. One sunny morning, she heard a tiny voice calling for help from near the sparkling stream.",
    },
    {
      filename: "scene_003.png",
      image_url:
        "https://images.unsplash.com/photo-1425082661705-1834bfd09dca?w=800&h=600&fit=crop",
      index: 3,
      text:
        "It was a little lost butterfly with beautiful rainbow wings! 'I can't find my way back to the flower garden,' said the butterfly sadly.",
    },
    {
      filename: "scene_004.png",
      image_url:
        "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800&h=600&fit=crop",
      index: 4,
      text:
        "Luna smiled kindly and said, 'Don't worry, I'll help you find your way home!' Together, they hopped and flew through the magical forest until they found the most beautiful garden full of colorful flowers. The butterfly was so happy and thanked Luna for being such a good friend!",
    },
  ],
  style: "whimsical children's book illustration",
};

