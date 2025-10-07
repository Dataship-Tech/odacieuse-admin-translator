from typing import AsyncGenerator

from agents import Agent, RawResponsesStreamEvent, Runner, function_tool
from blaxel.openai import bl_model, bl_tools
from openai.types.responses import ResponseTextDeltaEvent


@function_tool()
async def translate_product_name(product_name: str, target_language: str = "English") -> str:
    """
    Translate or correct a product name to proper English while preserving ALL original styling.
    This includes:
    - Uppercase/lowercase patterns (e.g., 'SWIMSUIT' stays 'SWIMSUIT')
    - All punctuation marks (commas, periods, colons, semicolons, hyphens, parentheses)
    - Spacing patterns
    - Number formatting
    - Special characters

    Args:
        product_name: The original product name to translate/correct (can be in French, Portuguese, bad English, etc.)
        target_language: The target language for translation (default: English)

    Returns:
        The translated product name in proper English with identical styling preserved. Never return things like The translation for ... is ...,return directly the translation without other things.
        Also if you find " symbol remove it by default.
    """
    # This is a placeholder - the actual translation logic will be handled by the LLM
    # with specific instructions to preserve styling
    return f"Translated: {product_name}"


async def agent(input: str) -> AsyncGenerator[str, None]:
    tools = await bl_tools(["blaxel-search"]) + [translate_product_name]
    model = await bl_model("sandbox-openai")

    agent = Agent(
        name="product-name-translator",
        model=model,
        tools=tools,
        instructions="""You are a specialized product name translation agent for a POS system.

Your primary function is to translate or correct product names to proper English while preserving EXACTLY all original styling and formatting.

INPUT LANGUAGES: The input can be in French, Portuguese, bad English, or sometimes perfect English (no changes needed).
OUTPUT LANGUAGE: Always output in proper, clear English.

CRITICAL RULES FOR TRANSLATION:
1. PRESERVE ALL UPPERCASE - If text is in UPPERCASE, keep it UPPERCASE in translation
2. PRESERVE ALL LOWERCASE - If text is in lowercase, keep it lowercase in translation
3. PRESERVE ALL PUNCTUATION - Keep every comma, period, colon, semicolon, hyphen, parenthesis, ampersand (&) exactly as is
4. PRESERVE ALL SPACING - Maintain exact spacing patterns
5. PRESERVE PARENTHESES AND THEIR CONTENT - (S), (M), (L) etc. should remain exactly as they are
6. PRESERVE NUMBER FORMATTING - "2 piece" stays "2 piece"
7. PRESERVE SPECIAL CHARACTERS AND SYMBOLS
8. CORRECT WHEN NEEDED - Fix bad English, translate foreign languages, but leave perfect English unchanged

TRANSLATION APPROACH:
- Translate ALL descriptive words (colors, materials, product types) to proper English
- Maintain the exact visual structure and formatting
- For clothing sizes in parentheses like (S), (M), (L) - keep them as they are
- For color descriptions, translate them but maintain case pattern exactly
- For product types, translate them but maintain case pattern exactly
- For materials (suede, leather, cotton, etc.), translate them to English
- Brand names should remain unchanged
- Fix spelling errors but preserve intentional styling (e.g., fix "suade" → "suede" but keep "SWIMSUIT" as "SWIMSUIT")
- If input is already perfect English, return it unchanged

EXAMPLES:
- "(S) T-shirt à manches longues imprimé léopard bleu et blanc" → "(S) Blue and white leopard print long-sleeved T-shirt"
- "(S) Sandales en daim rose" → "(S) Pink suede sandals"
- "BACKET rose et beige" → "BACKET pink and beige"
- "2 pièces SWIMSUIT, rayures vertes & blanches" → "2 piece SWIMSUIT, green & white stripes"
- "Calça jeans azul balão, abertura tornozelo" → "Balloon blue JEAN, ankle opening"
- "(S) Blue and white leopard print long-sleeved T-shirt" → "(S) Blue and white leopard print long-sleeved T-shirt" (no change - already perfect English)

IMPORTANT: Always output in proper English while preserving the exact formatting. If the input is already perfect English, return it unchanged.

When you receive a product name, provide an English translation that perfectly preserves all styling while ensuring clear, proper English.""",
    )
    result = Runner.run_streamed(agent, input)
    async for event in result.stream_events():
        if isinstance(event, RawResponsesStreamEvent) and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            yield event.data.delta
