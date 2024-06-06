#### Updates:
① Implement the EasyAnimate DiT video generation with Llama3 8B 6bit quantization LLM prompt
- ✅ [2024/06/06] ExLlamaV2 and EasyAnimate
U can contact me thr ![twitter_1](https://github.com/frankchieng/ComfyUI_Aniportrait/assets/130369523/27b4fcae-e50c-477d-86f4-dacf7fd052f4)[twitter](https://twitter.com/kurtqian) ![wechat_1](https://github.com/frankchieng/ComfyUI_Aniportrait/assets/130369523/b95cd0a2-4188-4eb3-b1de-5f6eeab71045) Weixin：GalaticKing

### Llama3 generate positive prompt directly chained with EasyAnimate
![llama3 generated prompt not modified](https://github.com/frankchieng/ComfyUI_llm_easyanimiate/blob/main/assets/easyanimate1.png)
[workflow](https://github.com/frankchieng/ComfyUI_llm_easyanimiate/blob/main/assets/easyanimate_llm_chain_workflow.json)

### Llama3 generate positive prompt first,then modify the prompts and output easyanimate videos
![llama3 generate prompt](https://github.com/frankchieng/ComfyUI_llm_easyanimiate/blob/main/assets/easyanimate2.png)
![llama3 generated prompt modified](https://github.com/frankchieng/ComfyUI_llm_easyanimiate/blob/main/assets/easyanimate3.png)
[workflow](https://github.com/frankchieng/ComfyUI_llm_easyanimiate/blob/main/assets/easyanimate_without_llm_chain_workflow.json)

### some test results
<table class="center">
<tr>
    <td width=100% style="border: none">
        <video controls autoplay loop src="https://github.com/frankchieng/ComfyUI_llm_easyanimiate/assets/130369523/86a20d32-702a-4a3f-a7d5-abcf1419a23c" muted="false"></video>
    </td>
</tr>
<tr>
    <td width=100% style="border: none">
        <video controls autoplay loop src="https://github.com/frankchieng/ComfyUI_llm_easyanimiate/assets/130369523/381d7458-57d9-4da4-8e47-a6aa32f39fad" muted="false"></video>
    </td>
</tr>
<tr>
    <td width=100% style="border: none">
        <video controls autoplay loop src="https://github.com/frankchieng/ComfyUI_llm_easyanimiate/assets/130369523/da7c3388-c17b-4b69-bbdf-51dbb4a12cbf" muted="false"></video>
    </td>
</tr>
</table>

EasyAnimate modules structure as below:
```text
./ComfyUI/
|-- models
|   |-- EasyAnimate
|   |   |-- Diffusion Transformer
|   |   |   |-- EasyAnimateV2-XL-2-768x768
|   |   |-- Personalizd_Model
|   |   |   |-- easyanimatev2_minimalism_lora.safetensors (you can put your own lora trained model here)
```
you can download model of [EasyAnimateV2-XL-2-768x768](https://huggingface.co/alibaba-pai/EasyAnimateV2-XL-2-768x768/tree/main)

(Lora of Pixart)[easyanimatev2_minimalism_lora.safetensors](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Personalized_Model/easyanimatev2_minimalism_lora.safetensors)
A lora training with a specifial type images. Images can be downloaded from [Url](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/Minimalism.zip).

Tips :
For better render performance,you'd better have A100GPU around 40G,i've been tested with a RTX4090,the maximum resolution is 736*512,otherwise will be OOM, current model video length can be reached 6 secs,144 frames with 24 fps
you have to install the [ComfyUI-ExLlama-Nodes](https://github.com/Zuellni/ComfyUI-ExLlama-Nodes) and [comfyui-mixlab-nodes](https://github.com/shadowcz007/comfyui-mixlab-nodes) custome code as well

To use a model with the nodes, you should clone its repository with git or manually download all the files and place them in ComfyUI/models/llm. For example, if you'd like to download the 6-bit [Llama-3-8B-Instruct](https://huggingface.co/turboderp/Llama-3-8B-Instruct-exl2/tree/6.0bpw), use the following command:
```shell
git install lfs
git clone https://huggingface.co/turboderp/Llama-3-8B-Instruct-exl2 -b 6.0bpw models/llm/Llama-3-8B-Instruct-exl2-6.0bpw
```

play with the llama3 instruction template like the panel as below:
```text
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Come up with a detailed unique Stable Diffusion prompt,the prompt should always begin with like this:This video shows or The video features<|eot_id|><|start_header_id|>assistant<|end_header_id|>

This video shows the majestic beauty of a waterfall cascading down a cliff into a serene lake.The waterfall,with its powerful flow,is the central focus of the video.The surrounding landscape is lush and green,with trees and folige adding to the natural beauty of the scene.<|eot_id|><|start_header_id|>user<|end_header_id|>

How about another one?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The video features a young woman with black eyes and blonde hair standing in a forrest wearing a crown.She seems to be lost in thought,and the camera focuses on her face.The atmosphere is serene,adn the shot is in slow motion.The video is of high quality,and the view is very clear.High quality,masterpiece,best quality,highres,ultra-detailed,fantastic.<|eot_id|><|start_header_id|>user<|end_header_id|>

only generate one more,tulips<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```
you can change the subject of prompt from tulips to flowers, fire, man,woman or whatever
