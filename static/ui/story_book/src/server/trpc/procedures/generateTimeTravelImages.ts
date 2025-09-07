import { z } from "zod";
import { TRPCError } from "@trpc/server";
import { baseProcedure } from "~/server/trpc/main";
import { experimental_generateImage as generateImage } from "ai";
import { openai } from "@ai-sdk/openai";
import { env } from "~/server/env";

const timeTravelImageSchema = z.object({
  id: z.string(),
  imageBase64: z.string(),
  era: z.string(),
  year: z.string(), 
  description: z.string(),
});

const inputSchema = z.object({
  cityName: z.string().min(1, "City name cannot be empty").max(100, "City name too long"),
});

export const generateTimeTravelImages = baseProcedure
  .input(inputSchema)
  .mutation(async ({ input }) => {
    console.log("Generating time travel images for city:", input.cityName);

    try {
      // Validate API key
      if (!env.OPENAI_API_KEY) {
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "OpenAI API key not configured",
        });
      }

      // Validate API key format
      if (!env.OPENAI_API_KEY.startsWith('sk-')) {
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Invalid OpenAI API key format",
        });
      }

      const model = openai.image("dall-e-3");
      
      // Define the historical eras we want to generate
      const eras = [
        {
          id: "1",
          era: "Ancient Times",
          year: "500 BC",
          description: `Ancient ${input.cityName} with historical architecture and early settlements`,
          promptDetails: "ancient civilization, stone buildings, early settlements, historical accuracy, archaeological style"
        },
        {
          id: "2", 
          era: "Medieval Period",
          year: "1200 AD",
          description: `Medieval ${input.cityName} showing castles, markets, and period-appropriate buildings`,
          promptDetails: "medieval architecture, castles, cobblestone streets, market squares, gothic style, period-appropriate clothing"
        },
        {
          id: "3",
          era: "Industrial Revolution",
          year: "1850 AD",
          description: `${input.cityName} during the industrial boom with factories and steam-powered transportation`,
          promptDetails: "industrial revolution era, steam engines, factory smokestacks, brick buildings, Victorian architecture, bustling activity"
        },
        {
          id: "4",
          era: "Modern Era", 
          year: "2020 AD",
          description: `Contemporary ${input.cityName} with modern skyline and current architecture`,
          promptDetails: "modern city skyline, glass buildings, contemporary architecture, urban landscape, current day"
        }
      ];

      console.log(`Generating ${eras.length} time travel images for ${input.cityName}...`);
      
      // Generate images for each era
      const generatedImages = [];
      
      for (const era of eras) {
        console.log(`Generating image for ${era.era} (${era.year})...`);
        
        // Construct prompt for this specific era
        const prompt = `A detailed, historically accurate illustration of ${input.cityName} in ${era.era} (${era.year}). Show ${era.promptDetails}. High quality, realistic style, aerial or street view perspective. Professional historical reconstruction artwork.`;
        
        // Truncate prompt if too long
        const finalPrompt = prompt.length > 1000 ? prompt.substring(0, 1000) : prompt;
        
        try {
          const { image } = await generateImage({
            model,
            prompt: finalPrompt,
            size: "1024x1024",
          });

          generatedImages.push({
            id: era.id,
            imageBase64: image.base64,
            era: era.era,
            year: era.year,
            description: era.description,
          });
          
          console.log(`Successfully generated image for ${era.era}`);
          
          // Add a small delay between requests to be respectful to the API
          if (generatedImages.length < eras.length) {
            await new Promise(resolve => setTimeout(resolve, 1000));
          }
        } catch (eraError: any) {
          console.error(`Failed to generate image for ${era.era}:`, eraError.message);
          // Continue with other eras even if one fails
          generatedImages.push({
            id: era.id,
            imageBase64: "", // Empty base64 for failed generation
            era: era.era,
            year: era.year,
            description: `${era.description} (Generation failed)`,
          });
        }
      }

      console.log(`Successfully generated ${generatedImages.filter(img => img.imageBase64).length}/${eras.length} images`);

      return {
        success: true,
        cityName: input.cityName,
        images: generatedImages,
      };

    } catch (error: any) {
      console.error("Error generating time travel images:", {
        error: error.message,
        cityName: input.cityName,
      });

      // Handle specific error cases
      let errorMessage = "Failed to generate time travel images";
      
      if (error.message?.includes("billing")) {
        errorMessage = "OpenAI billing limit reached. Please check your OpenAI account billing status.";
      } else if (error.message?.includes("quota")) {
        errorMessage = "OpenAI API quota exceeded. Please try again later or check your usage limits.";
      } else if (error.message?.includes("rate limit")) {
        errorMessage = "Rate limit exceeded. Please wait a moment and try again.";
      } else if (error.message?.includes("invalid_api_key")) {
        errorMessage = "Invalid OpenAI API key. Please check your API key configuration.";
      } else if (error.message?.includes("model_not_found")) {
        errorMessage = "DALL-E 3 model not available. Your API key may not have access to image generation.";
      } else if (error.message) {
        errorMessage = `OpenAI API Error: ${error.message}`;
      }
      
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message: errorMessage,
      });
    }
  });
