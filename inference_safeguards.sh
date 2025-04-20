# huggingface-cli login --token $1 --add-to-git-credential

echo "Evaluating ShieldGemma..."
sh inference_safeguard.sh ShieldGemma google/shieldgemma-9b ShieldGemma > inference_safeguard_shieldgemma.out

echo "Evaluating PolyGuard..."
sh inference_safeguard.sh PolyGuard ToxicityPrompts/PolyGuard-Qwen-Smol PolyGuard > inference_safeguard_polyguard.out

echo "Evaluating LlamaGuard..."
sh inference_safeguard.sh LlamaGuard meta-llama/Llama-Guard-3-8B LlamaGuard > inference_safeguard_llamaguard.out

echo "Evaluating WildGuard..."
sh inference_safeguard.sh WildGuard allenai/wildguard WildGuard > inference_safeguard_wildguard.out
sh multigpu_inference_safeguard.sh WildGuard allenai/wildguard WildGuard > inference_safeguard_wildguard.out

echo "Evaluating Llama-SealionGuard..."
sh inference_safeguard.sh SealionGuard aisingapore/Llama-SEA-LION-Guard Llama-SealionGuard > inference_safeguard_llama_sealionguard.out

echo "Evaluating Gemma-SealionGuard..."
sh inference_safeguard.sh SealionGuard aisingapore/Gemma-SEA-LION-Guard Gemma-SealionGuard > inference_safeguard_gemma_sealionguard.out

echo "Evaluating Llama-SealionGuard-EN-Cultural..."
sh inference_safeguard.sh SealionGuard aisingapore/Llama-SEA-LION-Guard-EN-Cultural Llama-SealionGuard-EN-Cultural > inference_safeguard_llama_sealionguard_en_cultural.out

echo "Evaluating Llama-SEA-LION-Guard-THEN-Cultural..."
sh inference_safeguard.sh SealionGuard aisingapore/Llama-SEA-LION-Guard-THEN-Cultural Llama-SEA-LION-Guard-THEN-Cultural > inference_safeguard_llama_sealionguard_then_cultural.out

echo "Evaluating Llama-SEA-LION-Guard-TH-Cultural..."
sh inference_safeguard.sh SealionGuard aisingapore/Llama-SEA-LION-Guard-TH-Cultural Llama-SEA-LION-Guard-TH-Cultural > inference_safeguard_llama_sealionguard_th_cultural.out
sh multigpu_inference_safeguard.sh SealionGuard aisingapore/Llama-SEA-LION-Guard-TH-Cultural Llama-SEA-LION-Guard-TH-Cultural > inference_safeguard_llama_sealionguard_th_cultural.out