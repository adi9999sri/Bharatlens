# BharatLens — Telugu news annotation guidelines (v0.1)

## Labeling unit
One headline per row. Assign **all** relevant labels based on what the headline explicitly says. Do not infer facts absent from the text. `Other` means all five binary labels are zero.

## Domain definitions

| Label | Include | Exclude |
|---|---|---|
| environment | Pollution, biodiversity, forests, ecology, emissions and environmental protection | Generic business or energy trade news without an ecological implication |
| culture | Languages, traditions, heritage, archaeology, folk arts and historic preservation | Generic celebrity or entertainment coverage |
| education | Schools, colleges, universities, learning, education policy, teachers and students' academic activities | Science or space news with no educational connection |
| agriculture | Farmers, cultivation, crops, soil management, agricultural livelihoods and farm markets | Generic food retail unrelated to farming |
| water | Drinking water, groundwater, rivers, reservoirs, irrigation, water governance, floods and water conservation | Ordinary weather forecasts without a substantive water-related impact |

## Multi-label decisions
- Lake pollution: environment + water. Add agriculture only if impacts on farmers or farming are stated.
- Irrigation allocation to crops: agriculture + water.
- School students researching river biodiversity: education + environment + water.
- Traditional seed preservation by farmers: culture + agriculture when the cultural tradition is explicit.

## Annotation procedure
1. Read the exact text provided, not just a URL slug.
2. Mark each of the five labels 0 or 1 independently.
3. If no category applies, leave all five as 0.
4. Do not force a category when evidence is insufficient; record uncertainties separately in a future review sheet.
5. Have a second annotator review ambiguous items before creating a held-out test set.
6. Deduplicate near-identical stories before any train/validation/test split.

## Known limitations
Headlines often omit context. Human annotation is provisional if full permitted text isn't available. Synthetic examples are useful for pipeline testing but **not** robust evaluation on real journalism.
