# Word Prediction Service Using Optimized Small Language Models (LLMs)

Addidtional libraries:
- torch
- transformers
- bitsandbytes
- accelerate
- pip install -U scikit-learn
- pip install datasets


# Documentation:
https://docs.google.com/document/d/1iFZkIRQ7qxilfz0paxmxlEf2yeEWH8J9lVoRb3sGOBg/edit?usp=sharing




# Word prediction service based on LLM
Create a small service based on Hugging Face Transfomers, that is focused on high performance (throughput) and low latency (responsiveness). The project should use a possibly small LLM to ensure satisfactory results. To this end, it is necessary to develop an appropriate evaluation metric, such as perplexity or accuracy, to measure the quality of the model's performance.


1. Presentation of the application prototype (16.12.2024 - 20.12.2024): present a working prototype of the service with basic functionalities.
1. Presentation of the finished application (20.01.2025 - 24.01.2025): present the finished application, including tests to validate the quality of the service
1. Project report (27.01.2025 - 30.01.2025): describe test results, how the application works, how to run it, and the scalability and reliability of the service 

Uploading to enauczanie.pg.edu.pl the code and project report. The report should include:
- A description of how the service works
- A description of the libraries and data used
- A description of how the service runs
- A description of the scalability and reliability of the service.
- Quality test results

Passing criteria
· Each stage must be completed and presented no later than the specified date.
· Attendance of all group members at stage presentation meetings is mandatory. Unexcused absence will result in failure of the stage by the absent student.
· The final grade for the project is the average of the grades for each stage (stages 2 through 5).




# model_name='distilgpt2'

# llama 3.2B

# metrics for model na eval
# perplexity
# bleu score
# czas generowania

# optymalzacja
# batchowanie
# flash attenction (jak z kwantyzacja)
# torchscript


# cashwanie promptow hugging face kv- cashe'ing






For test and development
$~ uvicorn src.main:app --reload



Trash list:
ValueError: Unrecognized model in tensorblock/TinyLlama-1.1B-intermediate-step-1431k-3T-GGUF. Should have a `model_type` key in its config.json, or contain one of the following strings in its name: albert, align, altclip, audio-spectrogram-transformer, autoformer, bark, bart, beit, bert, bert-generation, big_bird, bigbird_pegasus, biogpt, bit, blenderbot, blenderbot-small, blip, blip-2, bloom, bridgetower, bros, camembert, canine, chameleon, chinese_clip, chinese_clip_vision_model, clap, clip, clip_text_model, clip_vision_model, clipseg, clvp, code_llama, codegen, cohere, conditional_detr, convbert, convnext, convnextv2, cpmant, ctrl, cvt, dac, data2vec-audio, data2vec-text, data2vec-vision, dbrx, deberta, deberta-v2, decision_transformer, deformable_detr, deit, depth_anything, deta, detr, dinat, dinov2, distilbert, donut-swin, dpr, dpt, efficientformer, efficientnet, electra, encodec, encoder-decoder, ernie, ernie_m, esm, falcon, falcon_mamba, fastspeech2_conformer, flaubert, flava, fnet, focalnet, fsmt, funnel, fuyu, gemma, gemma2, git, glm, glpn, gpt-sw3, gpt2, gpt_bigcode, gpt_neo, gpt_neox, gpt_neox_japanese, gptj, gptsan-japanese, granite, granitemoe, graphormer, grounding-dino, groupvit, hiera, hubert, ibert, idefics, idefics2, idefics3, imagegpt, informer, instructblip, instructblipvideo, jamba, jetmoe, jukebox, kosmos-2, layoutlm, layoutlmv2, layoutlmv3, led, levit, lilt, llama, llava, llava_next, llava_next_video, llava_onevision, longformer, longt5, luke, lxmert, m2m_100, mamba, mamba2, marian, markuplm, mask2former, maskformer, maskformer-swin, mbart, mctct, mega, megatron-bert, mgp-str, mimi, mistral, mixtral, mllama, mobilebert, mobilenet_v1, mobilenet_v2, mobilevit, mobilevitv2, moshi, mpnet, mpt, mra, mt5, musicgen, musicgen_melody, mvp, nat, nemotron, nezha, nllb-moe, nougat, nystromformer, olmo, olmoe, omdet-turbo, oneformer, open-llama, openai-gpt, opt, owlv2, owlvit, paligemma, patchtsmixer, patchtst, pegasus, pegasus_x, perceiver, persimmon, phi, phi3, phimoe, pix2struct, pixtral, plbart, poolformer, pop2piano, prophetnet, pvt, pvt_v2, qdqbert, qwen2, qwen2_audio, qwen2_audio_encoder, qwen2_moe, qwen2_vl, rag, realm, recurrent_gemma, reformer, regnet, rembert, resnet, retribert, roberta, roberta-prelayernorm, roc_bert, roformer, rt_detr, rt_detr_resnet, rwkv, sam, seamless_m4t, seamless_m4t_v2, segformer, seggpt, sew, sew-d, siglip, siglip_vision_model, speech-encoder-decoder, speech_to_text, speech_to_text_2, speecht5, splinter, squeezebert, stablelm, starcoder2, superpoint, swiftformer, swin, swin2sr, swinv2, switch_transformers, t5, table-transformer, tapas, time_series_transformer, timesformer, timm_backbone, trajectory_transformer, transfo-xl, trocr, tvlt, tvp, udop, umt5, unispeech, unispeech-sat, univnet, upernet, van, video_llava, videomae, vilt, vipllava, vision-encoder-decoder, vision-text-dual-encoder, visual_bert, vit, vit_hybrid, vit_mae, vit_msn, vitdet, vitmatte, vits, vivit, wav2vec2, wav2vec2-bert, wav2vec2-conformer, wavlm, whisper, xclip, xglm, xlm, xlm-prophetnet, xlm-roberta, xlm-roberta-xl, xlnet, xmod, yolos, yoso, zamba, zoedepth







Prompts:
Long mathematical equ
	
Response body

{
  "text": "Long mathematical equ",
  "predictions": [
    "Long",
    "mathematical",
    "equ",
    "ution",
    "problems",
    "and",
    "answers",
    "It",
    "is",
    "the",
    "ratio",
    "of",
    "circumference",
    "to",
    "diameter.",
    "It",
    "can",
    "be",
    "used",
    "for",
    "any"
  ]
}