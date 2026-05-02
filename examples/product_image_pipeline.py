"""
=============================================================================
Example — Product Image Pipeline
=============================================================================

WHAT THIS EXAMPLE BUILDS
--------------------------
An agent that takes a product photo, removes its background, generates a
targeted background scene using DALL-E 3, and composites the product onto
it. The LLM handles the creative direction — you tell it the product and
the target audience; it decides what the scene should look like.

Pipeline:
    product.jpg
        |
        v
    remove_background()       <- rembg (local, no API key needed)
        |
        v
    product_nobg.png
        |
        v
    [Claude reasons: what background suits this audience?]
        |
        v
    generate_background()     <- DALL-E 3 (OpenAI API key needed)
        |
        v
    background.png
        |
        v
    composite_images()        <- Pillow (local)
        |
        v
    final_product.png


REQUIREMENTS
------------
    pip install rembg Pillow openai requests

    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...

RUNNING
-------
    python examples/product_image_pipeline.py --image path/to/product.jpg \
        --audience "Gen-Z fitness enthusiasts, 18-25, urban"

    python examples/product_image_pipeline.py --image path/to/sneakers.jpg \
        --audience "luxury fashion buyers, premium lifestyle"

OUTPUT
------
    output/product_nobg.png      — product with transparent background
    output/background.png        — generated background scene
    output/final_product.png     — final composited image
=============================================================================
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import requests
from fastmcp import FastMCP
from PIL import Image

from mcp_agent_framework import AgentConfig, AnthropicClient
from mcp_agent_framework.patterns import SingleAgentLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Output directory for all generated files
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# MCP server — three tools the agent will orchestrate
# ------------------------------------------------------------------
app = FastMCP("product-image-pipeline")


@app.tool
async def remove_background(image_path: str) -> str:
    """
    Remove the background from a product image.

    Uses the rembg library (U2Net model) to isolate the product.
    Returns the path to a PNG file with a transparent background.
    The model runs locally — no API call, no cost.

    Args:
        image_path: Path to the input product image (JPG, PNG, WEBP).

    Returns:
        Path to the output PNG with transparent background.
    """
    try:
        from rembg import remove
    except ImportError:
        return "Error: rembg not installed. Run: pip install rembg"

    input_path = Path(image_path)
    if not input_path.exists():
        return f"Error: file not found: {image_path}"

    output_path = OUTPUT_DIR / f"{input_path.stem}_nobg.png"

    logger.info("Removing background from: %s", input_path)

    with open(input_path, "rb") as f:
        input_data = f.read()

    output_data = remove(input_data)

    with open(output_path, "wb") as f:
        f.write(output_data)

    # Report product dimensions so the agent can reason about compositing
    img = Image.open(output_path)
    w, h = img.size
    logger.info("Background removed. Output: %s (%dx%d)", output_path, w, h)

    return f"Background removed successfully. Product PNG saved to: {output_path} (size: {w}x{h}px)"


@app.tool
async def generate_background(
    prompt: str,
    size: str = "1024x1024",
) -> str:
    """
    Generate a background scene image using DALL-E 3.

    Craft a detailed visual prompt describing the scene, lighting, mood,
    and environment. The more specific the prompt, the better the result.

    Args:
        prompt: Detailed description of the background scene.
                Example: "Modern minimalist gym interior, polished concrete
                floors, dramatic neon accent lighting in blue and purple,
                no people, wide angle, product photography backdrop"
        size:   Image dimensions. Options: "1024x1024", "1792x1024", "1024x1792".
                Use "1792x1024" for landscape/banner, "1024x1024" for square.

    Returns:
        Path to the downloaded background image.
    """
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY environment variable not set."

    valid_sizes = {"1024x1024", "1792x1024", "1024x1792"}
    if size not in valid_sizes:
        size = "1024x1024"

    logger.info("Generating background with DALL-E 3...")
    logger.info("Prompt: %s", prompt[:120])

    client = openai.OpenAI(api_key=api_key)

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    revised_prompt = response.data[0].revised_prompt

    # Download the generated image
    img_response = requests.get(image_url, timeout=30)
    img_response.raise_for_status()

    output_path = OUTPUT_DIR / "background.png"
    with open(output_path, "wb") as f:
        f.write(img_response.content)

    img = Image.open(output_path)
    w, h = img.size
    logger.info("Background generated. Output: %s (%dx%d)", output_path, w, h)

    return (
        f"Background generated successfully.\n"
        f"Saved to: {output_path} (size: {w}x{h}px)\n"
        f"DALL-E revised prompt: {revised_prompt[:200]}"
    )


@app.tool
async def composite_images(
    product_path: str,
    background_path: str,
    product_scale: float = 0.55,
    vertical_position: str = "center",
) -> str:
    """
    Composite a product (transparent PNG) onto a background image.

    The product is centered horizontally. Vertical position is controlled
    by the vertical_position parameter. A subtle drop shadow is added to
    ground the product in the scene.

    Args:
        product_path:      Path to product PNG with transparent background.
        background_path:   Path to background image.
        product_scale:     Product width as a fraction of background width.
                           0.4 = 40% of background width (small, environmental).
                           0.6 = 60% (prominent, hero shot).
                           0.8 = 80% (very large, close-up feel).
        vertical_position: Where to place the product vertically.
                           "center"        - true center of frame.
                           "lower-center"  - slightly below center (natural for most products).
                           "bottom"        - near bottom (good for shoes, ground products).

    Returns:
        Path to the final composited image.
    """
    product_p = Path(product_path)
    background_p = Path(background_path)

    if not product_p.exists():
        return f"Error: product file not found: {product_path}"
    if not background_p.exists():
        return f"Error: background file not found: {background_path}"

    product    = Image.open(product_p).convert("RGBA")
    background = Image.open(background_p).convert("RGBA")

    bg_w, bg_h = background.size

    # Scale product to requested fraction of background width
    new_w = int(bg_w * product_scale)
    ratio = new_w / product.width
    new_h = int(product.height * ratio)
    product = product.resize((new_w, new_h), Image.LANCZOS)

    # Horizontal: always centered
    x = (bg_w - new_w) // 2

    # Vertical position
    if vertical_position == "bottom":
        y = bg_h - new_h - int(bg_h * 0.05)
    elif vertical_position == "lower-center":
        y = int((bg_h - new_h) * 0.6)
    else:  # center
        y = (bg_h - new_h) // 2

    # Add a subtle drop shadow to ground the product
    shadow = Image.new("RGBA", background.size, (0, 0, 0, 0))
    shadow_layer = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 60))
    shadow.paste(shadow_layer, (x + 8, y + 8), product)

    # Composite: background -> shadow -> product
    result = background.copy()
    result = Image.alpha_composite(result, shadow)
    result.paste(product, (x, y), product)

    # Save as high-quality PNG
    output_path = OUTPUT_DIR / "final_product.png"
    result.save(output_path, "PNG", optimize=True)

    logger.info("Composited. Output: %s", output_path)
    return (
        f"Compositing complete!\n"
        f"Final image saved to: {output_path}\n"
        f"Product placed at ({x}, {y}) on {bg_w}x{bg_h} background."
    )


# ------------------------------------------------------------------
# System prompt — guides the LLM's creative direction
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a product photography automation specialist.

Your job is to create a compelling, targeted product image by:
1. Removing the product background using remove_background()
2. Designing and generating an appropriate scene using generate_background()
3. Compositing the product onto the scene using composite_images()

CREATIVE DIRECTION GUIDELINES:

When crafting the background prompt, be highly specific. Include:
- The physical environment (indoor/outdoor, specific location)
- Lighting style (soft diffused, dramatic side-light, golden hour, studio)
- Color palette and mood (warm, cool, neutral, vibrant, muted)
- Surface the product appears to rest on or near
- The word "no people" to keep focus on the product
- "product photography backdrop" or "commercial photography" for quality

AUDIENCE-TO-SCENE MAPPING (examples):
- Gen-Z / youth / streetwear    -> urban concrete, neon accents, graffiti walls, skate parks
- Luxury / premium / high-end   -> marble surfaces, soft natural light, minimal negative space
- Outdoor / adventure / sports  -> mountain trails, forests, golden hour, dramatic skies
- Tech / innovation / startup   -> clean white desk, soft shadows, minimal distractions
- Wellness / fitness / health   -> gym interiors, yoga studios, fresh natural light
- Home / lifestyle / family     -> warm living rooms, kitchen counters, cozy natural light

For vertical_position in composite_images():
- Shoes, bags, bottles -> "bottom" (sits on ground naturally)
- Electronics, watches -> "lower-center" (floating, professional)
- Food, cosmetics      -> "center" (hero shot)

Always pick product_scale between 0.45 and 0.70 based on how prominent the
product should feel. Luxury = smaller (more space). Mass market = larger.

Report what creative choices you made and why."""


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
async def run_pipeline(image_path: str, audience: str) -> str:
    agent = SingleAgentLoop(
        llm_client=AnthropicClient("claude-sonnet-4-6"),
        config=AgentConfig(
            mcp_server_config=app,
            system_prompt=SYSTEM_PROMPT,
            max_iterations=10,
        ),
    )

    prompt = f"""
Product image path: {image_path}
Target audience: {audience}

Create a compelling product photo targeted at this audience.
Walk me through your creative reasoning, then execute the pipeline.
"""

    print("\n" + "=" * 60)
    print("PRODUCT IMAGE PIPELINE")
    print("=" * 60)
    print(f"Product : {image_path}")
    print(f"Audience: {audience}")
    print("=" * 60 + "\n")

    result = await SingleAgentLoop(
        llm_client=AnthropicClient("claude-sonnet-4-6"),
        config=AgentConfig(
            mcp_server_config=app,
            system_prompt=SYSTEM_PROMPT,
            max_iterations=10,
        ),
    ).run(prompt)

    print("\nAGENT OUTPUT:")
    print("-" * 60)
    print(result)
    print("-" * 60)
    print(f"\nOutputs written to: {OUTPUT_DIR.absolute()}/")

    return result


def main():
    parser = argparse.ArgumentParser(description="AI Product Image Pipeline")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the product image (JPG, PNG, WEBP)",
    )
    parser.add_argument(
        "--audience",
        required=True,
        help='Target audience description. Example: "Gen-Z fitness enthusiasts, 18-25, urban"',
    )
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    asyncio.run(run_pipeline(args.image, args.audience))


if __name__ == "__main__":
    main()
