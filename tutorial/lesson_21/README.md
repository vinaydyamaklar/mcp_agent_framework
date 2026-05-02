# Lesson 21 — Multi-Modal Pipeline: Product Image Automation

**Unit 7: Production Infrastructure**

---

## What you will learn

- How to combine LLMs with non-language models (image generation, image processing)
- How external APIs and local libraries become MCP tools
- How to write a system prompt that guides creative decision-making
- The pattern: LLM reasons, tools execute

---

## The concept

Every lesson so far used LLMs for text. The real world has more modalities — images, audio, video, documents. The framework handles them the same way: wrap them as MCP tools. The LLM coordinates; the tools do the work.

This lesson builds a product image pipeline:

```
product.jpg
    |
    v
remove_background()     <- rembg (local Python library, free)
    |
    v
product_nobg.png
    |
    v
[Claude decides: what scene fits this audience?]
    |
    v
generate_background()   <- DALL-E 3 (OpenAI image model)
    |
    v
background.png
    |
    v
composite_images()      <- Pillow (local Python library, free)
    |
    v
final_product.png
```

Three tools. One agent. The LLM does none of the pixel work — it does the creative direction.

---

## Why this architecture works

**The LLM is good at:**
- Understanding "Gen-Z fitness enthusiasts, 18-25, urban streetwear"
- Translating that into a specific DALL-E prompt: "urban concrete skate park, neon accent lighting, spray paint textures..."
- Deciding product scale (luxury = smaller, mass market = larger)
- Deciding vertical position (shoes go bottom, watches go center)

**The LLM is bad at:**
- Removing image backgrounds
- Generating images
- Pixel compositing

So you give the hard visual work to tools built for it, and let the LLM handle judgment.

---

## The three tools

### `remove_background(image_path)`

Uses `rembg` — a local library running a U2Net neural network. No API key, no cost, runs offline. Returns a PNG with a transparent background.

```python
@app.tool
async def remove_background(image_path: str) -> str:
    from rembg import remove
    output_data = remove(open(image_path, "rb").read())
    # save to output/product_nobg.png
    return f"Saved to: output/product_nobg.png"
```

### `generate_background(prompt, size)`

Calls DALL-E 3 via the OpenAI SDK. The LLM crafts the prompt — specific, detailed, audience-aware. Returns path to the downloaded image.

```python
@app.tool
async def generate_background(prompt: str, size: str = "1024x1024") -> str:
    client = openai.OpenAI()
    response = client.images.generate(model="dall-e-3", prompt=prompt, size=size)
    # download image, save to output/background.png
    return f"Saved to: output/background.png"
```

### `composite_images(product_path, background_path, product_scale, vertical_position)`

Pure Pillow. Resizes the product, adds a drop shadow, pastes it onto the background. The LLM picks `product_scale` (0.4-0.8) and `vertical_position` (center / lower-center / bottom) based on the product type.

```python
@app.tool
async def composite_images(product_path, background_path, product_scale=0.55, vertical_position="center") -> str:
    product    = Image.open(product_path).convert("RGBA")
    background = Image.open(background_path).convert("RGBA")
    # resize, position, add shadow, composite
    result.save("output/final_product.png")
    return "Saved to: output/final_product.png"
```

---

## The system prompt does the creative work

The system prompt doesn't just describe what to do — it encodes domain knowledge:

```python
SYSTEM_PROMPT = """
AUDIENCE-TO-SCENE MAPPING:
- Gen-Z / youth / streetwear    -> urban concrete, neon accents, graffiti walls
- Luxury / premium              -> marble surfaces, soft natural light, minimal space
- Outdoor / adventure / sports  -> mountain trails, forests, golden hour
- Tech / innovation             -> clean white desk, soft shadows, minimal

For vertical_position:
- Shoes, bags, bottles -> "bottom"
- Electronics, watches -> "lower-center"
- Food, cosmetics      -> "center"

Luxury = smaller product scale. Mass market = larger.
"""
```

This is how you make an LLM make good creative decisions consistently — not by hoping it figures it out, but by encoding the rules in the prompt.

---

## What this lesson demonstrates from the full curriculum

| This lesson uses | From lesson |
|---|---|
| `SingleAgentLoop` | L5 |
| MCP tools with `@app.tool` | L4 |
| System prompt as domain encoder | L3 |
| External API as tool (DALL-E) | L6 |
| Multi-model coordination | L8 (Orchestrator concept) |

This is the curriculum coming full circle. One agent, three tools — but the tools span a local ML model, a commercial image API, and a local graphics library. The framework treats them identically.

---

## Read this file

```
examples/product_image_pipeline.py
```

Read it top to bottom. Note:
- How each `@app.tool` function handles its own error checking
- How the system prompt encodes audience-to-scene mapping rules
- How `composite_images` adds a drop shadow to ground the product naturally
- How the agent is given `max_iterations=10` to allow the full 3-tool pipeline

---

## Run this

```bash
pip install rembg Pillow openai requests
export OPENAI_API_KEY=sk-...

python examples/product_image_pipeline.py \
  --image path/to/sneakers.jpg \
  --audience "Gen-Z fitness enthusiasts, 18-25, urban streetwear"
```

Check `output/` for three files:
1. `product_nobg.png` — transparent product
2. `background.png` — DALL-E generated scene
3. `final_product.png` — final composited image

---

## Exercise

Extend the pipeline with a fourth tool:

```python
@app.tool
async def add_text_overlay(
    image_path: str,
    headline: str,
    tagline: str,
    position: str = "bottom",
) -> str:
    """Add a headline and tagline to the final product image."""
    ...
```

The LLM should decide the headline and tagline based on the audience. For a luxury watch targeting premium buyers, it might generate: "Precision. Redefined." — not because you told it to, but because the audience description implies it.

---

## Key terms

| Term | Meaning |
|---|---|
| Multi-modal | Combining different model types (text LLM + image model + local library) |
| `rembg` | Local background removal using U2Net neural network |
| DALL-E 3 | OpenAI's image generation model, called via API |
| Compositing | Layering images with alpha blending — product over background |
| Drop shadow | Semi-transparent offset copy of the product to create depth |
| Creative direction via prompt | Encoding domain rules in the system prompt so the LLM makes consistent decisions |

---

*Lesson 21 of 21 — Applied AI Engineering*
