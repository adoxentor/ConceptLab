import random
import sys
from copy import deepcopy, copy
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyrallis
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from accelerate.utils import set_seed

from kandinsky2 import Kandinsky2_1, load_utils
from training.dataset import ConceptDataset
from training.templates import object_templates_edits, LearnableProperties
from training.train_config import TrainConfig

matplotlib.use('Agg')  # Set the backend to non-interactive (Agg)


class Coach:
    def __init__(self, config: TrainConfig):
        self.cfg = config
        (self.cfg.output_dir / 'run_cfg.yaml').write_text(pyrallis.dump(self.cfg))
        (self.cfg.output_dir / 'run.sh').write_text(f'python {Path(__file__).name} {" ".join(sys.argv)}')
        if self.cfg.seed is not None:
            set_seed(self.cfg.seed)

        self.model = self.get_model()

        # Save EOS token_id and update tokenizers
        self.t2_eos_token_id = len(self.model.tokenizer2.encoder) - 1
        self.t2_place_token_id = self.update_tokenizer()

        # Set optimizer and gradient flow
        self.set_model_gradient_flow()
        self.optimizer = self.get_optimizer()
        self.train_dataloader = self.get_train_dataloader()

        # Save original embeddings from both models
        self.orig_t2_params = self.model.clip_model.token_embedding.weight.data.clone()
        self.weight_dtype = self.model.model.dtype

        # Load blip model if needed
        self.blip_processor, self.blip_model = self.load_blip_vlm()
        self.train_step = 0

    def get_model(self) -> Kandinsky2_1:
        model = load_utils.get_model(self.cfg.cache_root, self.cfg.device)
        return model

    def query_vlm(self, sampled_image) -> str:
        # Format the question template with the appropriate token
        question = self.cfg.vlm_question.format(token=self.cfg.positive_classes[0])

        if self.cfg.use_online_vlm:
            import replicate
            import base64
            import io

            # Convert PIL Image to base64
            buffered = io.BytesIO()
            sampled_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            data_uri = f"data:image/png;base64,{img_str}"

            # Use Replicate's BLIP-2 API
            output = replicate.run(
                "andreasjansson/blip-2:f677695e5e89f8b236e52ecd1d3f01beb44c34606419bcc19345e046d8f786f9",
                input={
                    "image": data_uri,
                    "question": question,
                }
            )

            negative_answer = output
        else:
            # Original local model inference code
            with torch.no_grad():
                inputs = self.blip_processor(sampled_image, question, return_tensors="pt").to(self.blip_model.device, torch.float16)
                out = self.blip_model.generate(**inputs)
                negative_answer = self.blip_processor.decode(out[0], skip_special_tokens=True)

        # Remove leading "a ", if exists
        negative_cls = negative_answer
        if negative_cls.startswith("a "):
            negative_cls = negative_cls[2:]

        print(f'Adding negative class: "{negative_cls}"')
        return negative_cls

    def save_images(self, save_dir: Path, save_prefix: str):
        # Format each prompt template with the placeholder token
        prompts = [prompt.format(token=self.cfg.placeholder_token) for prompt in self.cfg.generation_prompts]
        inference_seeds = self.cfg.inference_seeds
        images = []
        for prompt in prompts:
            images.extend([self.model.generate_text2img(prompt="",
                                                        img_prompt=prompt,
                                                        num_steps=50,
                                                        batch_size=1,
                                                        guidance_scale=7.5,
                                                        h=self.cfg.img_size,
                                                        w=self.cfg.img_size,
                                                        sampler="p_sampler",
                                                        prior_cf_scale=4,
                                                        prior_steps="5",
                                                        seed=inference_seeds[idx])[0]
                           for idx in range(len(inference_seeds))])

        gen_images = np.hstack([np.array(img) for img in images])
        Image.fromarray(gen_images).save(save_dir / f'{save_prefix}.jpeg')
        # We return the first output of set_b which will be optionally used by BLIP
        return images[0]

    def create_training_summary(self):
        """Creates a summary image combining all step images with their negative class captions"""
        if not self.cfg.live_negatives or len(self.cfg.negative_classes) == 0:
            return

        images_dir = self.cfg.images_root
        step_images = []
        negative_classes = []
        
        # Collect all step images and their corresponding negative classes
        for step in range(0, self.train_step + 1, self.cfg.log_image_frequency):
            image_path = images_dir / f'{step}_step_images.jpeg'
            if image_path.exists():
                step_images.append(np.array(Image.open(image_path)))
                if step > 0:  # Skip initial step as it has no negative class yet
                    class_idx = (step // self.cfg.log_image_frequency) - 1
                    if class_idx < len(self.cfg.negative_classes):
                        negative_classes.append(self.cfg.negative_classes[class_idx])
        
        if not step_images:
            return
            
        # Calculate dimensions for the combined image
        text_height = 30  # Height for text area
        img_height = step_images[0].shape[0]
        img_width = step_images[0].shape[1]
        total_height = (img_height + text_height) * len(step_images)
        
        # Create the combined image
        combined_image = np.ones((total_height, img_width, 3), dtype=np.uint8) * 255
        
        # Add each image and its caption
        for idx, img in enumerate(step_images):
            y_offset = idx * (img_height + text_height)
            
            # Add the image
            combined_image[y_offset + text_height:y_offset + text_height + img_height] = img
            
            # Add the caption
            if idx > 0:  # Skip caption for initial image
                pil_image = Image.fromarray(combined_image)
                draw = ImageDraw.Draw(pil_image)
                text = f"Added negative class: {negative_classes[idx-1]}"
                draw.text((10, y_offset + 5), text, fill=(0, 0, 0))
                combined_image = np.array(pil_image)
        
        # Save the final summary image
        Image.fromarray(combined_image).save(self.cfg.output_dir / 'training_summary.jpeg')

    def load_blip_vlm(self) -> Tuple[Optional[torch.nn.Module], Optional[torch.nn.Module]]:
        if self.cfg.live_negatives:
            if self.cfg.use_online_vlm:
                # Return None, None since we'll use the API directly in query_vlm
                return None, None
            else:
                # Original local model loading code
                from transformers import Blip2Processor, Blip2ForConditionalGeneration

                blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
                
                # If using CPU, load in FP32. Otherwise use FP16 on GPU
                if self.cfg.device == "cpu":
                    blip_model = Blip2ForConditionalGeneration.from_pretrained(
                        "Salesforce/blip2-flan-t5-xl",
                        device_map=self.cfg.device,
                        torch_dtype=torch.float32
                    )
                else:
                    blip_model = Blip2ForConditionalGeneration.from_pretrained(
                        "Salesforce/blip2-flan-t5-xl", 
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                    
                return blip_processor, blip_model
        else:
            return None, None

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = ConceptDataset(
            placeholder_token=self.cfg.placeholder_token,
            learnable_property=self.cfg.learnable_property
        )

        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.cfg.train_batch_size, shuffle=True, num_workers=self.cfg.num_workers
        )
        return train_dataloader

    def get_optimizer(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            params=self.model.clip_model.token_embedding.parameters(),
            lr=self.cfg.learning_rate * self.cfg.train_batch_size,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            weight_decay=self.cfg.adam_weight_decay,
            eps=self.cfg.adam_epsilon,
        )
        return optimizer

    def update_tokenizer(self) -> Tuple[int, int]:
        # Convert the initializer_token, placeholder_token to ids for tokenizer2
        # and add placeholder_token to tokenizer2
        t2p_index_to_add = len(self.model.tokenizer2.encoder)
        self.model.tokenizer2.encoder[self.cfg.placeholder_token] = t2p_index_to_add
        self.model.tokenizer2.decoder[t2p_index_to_add] = self.cfg.placeholder_token
        self.model.tokenizer2.cache[self.cfg.placeholder_token] = self.cfg.placeholder_token

        t2_place_token_id = self.model.tokenizer2.encode(self.cfg.placeholder_token)[0]
        t2_init_token_id = self.model.tokenizer2.encode(self.cfg.initializer_token)[0]

        old_vocab_size, t2_embed_size = self.model.clip_model.token_embedding.weight.shape

        # Create new embeddings
        # Copy old weights to the new embeddings and initialize new token
        new_embed = nn.Embedding(old_vocab_size + 1, t2_embed_size).to(self.cfg.device)
        new_embed.weight.data[:old_vocab_size, :] = self.model.clip_model.token_embedding.weight.data.clone()
        new_embed.weight.data[t2_place_token_id, :] = new_embed.weight.data[t2_init_token_id, :]

        self.model.clip_model.token_embedding = deepcopy(new_embed)
        return t2_place_token_id

    def set_model_gradient_flow(self):
        ## Freeze all except embeddings
        self.model.image_encoder.requires_grad_(False)
        self.model.model.requires_grad_(False)
        self.model.prior.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)

        self.model.clip_model.token_embedding.requires_grad_(True)
        self.model.clip_model.transformer.requires_grad_(False)

    @staticmethod
    def check_tokens_is_valid(model: Kandinsky2_1, placeholder_token: str, initializer_token: str):
        print("Check tokens...")
        if placeholder_token in model.tokenizer2.encoder:
            raise ValueError(f"Word {placeholder_token} exists in tokenizer2. Please select another word.")

        if initializer_token not in model.tokenizer2.encoder:
            raise ValueError(f"Word {initializer_token} doesn't exist in tokenizer2. Please select another word.")
        print("Selected tokens are correct")

    def generate_clip_emb(self,
                          prompt: str,
                          batch_size: int = 1,
                          prior_cf_scale: int = 1,
                          prior_steps: str = "5",
                          negative_prior_prompt: str = "",
                          apply_prior=True) -> torch.Tensor:
        prompts_batch = [prompt for _ in range(batch_size)]
        prior_cf_scales_batch = [prior_cf_scale] * len(prompts_batch)
        prior_cf_scales_batch = torch.tensor(prior_cf_scales_batch, device=self.model.device)
        max_txt_length = self.model.prior.model.text_ctx
        tok, mask = self.model.tokenizer2.padded_tokens_and_mask(
            prompts_batch, max_txt_length
        )
        cf_token, cf_mask = self.model.tokenizer2.padded_tokens_and_mask(
            [negative_prior_prompt], max_txt_length
        )
        if not (cf_token.shape == tok.shape):
            cf_token = cf_token.expand(tok.shape[0], -1)
            cf_mask = cf_mask.expand(tok.shape[0], -1)
        tok = torch.cat([tok, cf_token], dim=0)
        mask = torch.cat([mask, cf_mask], dim=0)
        tok, mask = tok.to(device=self.model.device), mask.to(device=self.model.device)

        x = self.model.clip_model.token_embedding(tok).type(self.model.clip_model.dtype)
        x = x + self.model.clip_model.positional_embedding.type(self.model.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND|
        x = self.model.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.clip_model.ln_final(x).type(self.model.clip_model.dtype)
        txt_feat_seq = x
        # Fix to allow adding new to tokens
        txt_feat = (x[torch.arange(x.shape[0]), tok.eq(self.t2_eos_token_id).nonzero()[:1,
                                                -1]] @ self.model.clip_model.text_projection)

        txt_feat, txt_feat_seq = txt_feat.float().to(self.model.device), txt_feat_seq.float().to(self.model.device)

        if apply_prior:
            img_feat = self.model.prior(
                txt_feat,
                txt_feat_seq,
                mask,
                prior_cf_scales_batch,
                timestep_respacing=prior_steps,
            )
            img_feat = img_feat.to(self.model.model_dtype)
        else:
            img_feat = None

        txt_feat = txt_feat.to(self.model.model_dtype)

        return txt_feat, img_feat

    def get_normed_embeds(self, text_prompts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            text_tokens = self.model.tokenizer1(
                text_prompts,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            _, text_embs = self.model.text_encoder(
                tokens=text_tokens['input_ids'].long().to(device=self.cfg.device),
                mask=text_tokens['attention_mask'].to(device=self.cfg.device),
            )
            text_embs_normed = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        return text_embs_normed

    def train(self):
        # Save initial state
        if self.train_step == 0:
            self.save_embeds(self.cfg.output_dir / "0_step_embeds.bin")
            
        sampled_image = self.save_images(save_dir=self.cfg.images_root, save_prefix=f'init_images')
        if self.cfg.live_negatives and len(self.cfg.negative_classes) == 0:
            live_negative = self.query_vlm(sampled_image)
            self.cfg.negative_classes.append(live_negative)
            # Save initial negative classes
            self.save_embeds(self.cfg.output_dir / "0_step_embeds.bin")
        elif self.cfg.gradual_negatives:
            random.shuffle(self.cfg.negative_classes)
            self.cfg.negative_pool = copy(self.cfg.negative_classes)
            self.cfg.negative_classes = [self.cfg.negative_pool.pop(0)]
            # Save initial negative classes
            self.save_embeds(self.cfg.output_dir / "0_step_embeds.bin")

        distances_log: List[Dict[str, float]] = []

        while self.train_step < self.cfg.steps:
            for sample_idx, batch in enumerate(self.train_dataloader):
                print(f'For step #{self.train_step}')
                txt_emb, image_emb = self.generate_clip_emb(
                    prompt=batch["text"][0],
                    batch_size=self.cfg.train_batch_size,
                    prior_cf_scale=4,
                    prior_steps="5",
                    negative_prior_prompt="",
                    apply_prior=not self.cfg.optimize_in_text_space
                )
                if self.cfg.optimize_in_text_space:
                    image_emb = txt_emb[:1]

                image_emb_norm = image_emb.norm(p=2, dim=-1, keepdim=True)
                image_emb_normed = image_emb / image_emb_norm

                # Calculate distances from classes
                distances_per_cls = {}
                if self.cfg.learnable_property == LearnableProperties.style:
                    assert len(self.cfg.positive_classes) == 1, "Style mode uses one placeholder positive class"
                    # For style mode we just use the prompt without specifying the style, so positive class is ignored
                    pos_prompts = [batch["template"][0].format(token='').replace('in the style of ', '')]
                else:
                    pos_prompts = [batch["template"][0].format(token=pos_word) for pos_word in
                                   self.cfg.positive_classes]
                pos_embeds = self.get_normed_embeds(pos_prompts)
                pos_cosine_sim = (pos_embeds.detach() @ image_emb_normed.T)

                # Add distances to log
                for pos_ind, pos_class in enumerate(self.cfg.positive_classes):
                    distances_per_cls[pos_class] = pos_cosine_sim[pos_ind].mean().item()

                # Restrict max cosine sim to optimize
                pos_cosine_sim = torch.min(pos_cosine_sim, torch.ones_like(pos_cosine_sim) * self.cfg.max_cosine_thr)

                # Calc positive loss
                pos_loss: torch.Tensor = 0
                for pos_ind, curr_pos_cosine_sim in enumerate(pos_cosine_sim):
                    pos_loss += self.cfg.positive_weights[pos_ind] * (1 - curr_pos_cosine_sim)
                max_pos_cosine, pos_max_ind = pos_cosine_sim.mean(dim=1).max(dim=0)
                print(f'\tpos_loss: {pos_loss.item():.3f}, '
                      f'max_pos: {max_pos_cosine:.3f} for {pos_prompts[pos_max_ind]} ')

                neg_prompts = [batch["template"][0].format(token=neg_word) for neg_word in self.cfg.negative_classes]
                if len(neg_prompts) > 0:
                    # Calc distances to negative classes
                    neg_embeds = self.get_normed_embeds(neg_prompts)
                    neg_cosine_sim = (neg_embeds.detach() @ image_emb_normed.T)

                    # Add distances to log
                    for neg_ind, neg_class in enumerate(self.cfg.negative_classes):
                        distances_per_cls[neg_class] = neg_cosine_sim[neg_ind].mean().item()

                    # Restrict min cosine sim to optimize
                    neg_cosine_sim = torch.max(neg_cosine_sim,
                                               torch.ones_like(neg_cosine_sim) * self.cfg.min_cosine_thr)

                    mean_neg_cosine = neg_cosine_sim.mean()
                    max_neg_cosine, neg_max_ind = neg_cosine_sim.mean(dim=1).max(dim=0)
                    print(f'\tmean_neg: {mean_neg_cosine:.3f}, '
                          f'max_neg: {max_neg_cosine:.3f} for {neg_prompts[neg_max_ind]} ')
                else:
                    mean_neg_cosine = 0
                    max_neg_cosine = 0

                distances_log.append(distances_per_cls)
                print(distances_per_cls.keys())

                loss = 0.5 * (mean_neg_cosine + max_neg_cosine) + self.cfg.pos_to_neg_loss_factor * pos_loss
                print(f'\tloss: {loss.item():.3f}')
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # We don't need update all embeddings weights. Only new embeddings.
                with torch.no_grad():
                    index_no_updates_t2 = torch.arange(
                        self.model.clip_model.token_embedding.weight.shape[0]) != self.t2_place_token_id
                    self.model.clip_model.token_embedding.weight[index_no_updates_t2] = \
                        self.orig_t2_params[index_no_updates_t2]

                self.train_step += 1

                if self.train_step % self.cfg.log_embed_frequency == 0:
                    embed_save_path = self.cfg.output_dir / f"{self.train_step}_step_embeds.bin"
                    self.save_embeds(embed_save_path)

                if self.cfg.log_image_frequency > 0 and (self.train_step % self.cfg.log_image_frequency == 0):
                    sampled_image = self.save_images(save_dir=self.cfg.images_root,
                                                     save_prefix=f'{self.train_step}_step_images')
                    figure_save_path = self.cfg.images_root / f"{self.train_step}_step_distances.jpg"
                    self.plot_distances(distances_log=distances_log, output_path=figure_save_path)

                    if self.cfg.live_negatives:
                        negative = self.query_vlm(sampled_image)
                        self.cfg.negative_classes.append(negative)
                        # Save updated negative classes list
                        with open(self.cfg.images_root / f'{self.train_step}_negative_classes.txt', 'w') as f:
                            f.write('\n'.join(self.cfg.negative_classes))
                    elif self.cfg.gradual_negatives:
                        if len(self.cfg.negative_pool) > 0:
                            self.cfg.negative_classes.append(self.cfg.negative_pool.pop(0))
                            # Save updated negative classes list
                            with open(self.cfg.images_root / f'{self.train_step}_negative_classes.txt', 'w') as f:
                                f.write('\n'.join(self.cfg.negative_classes))

            embed_save_path = self.cfg.output_dir / "learned_embeds.bin"
            self.save_embeds(embed_save_path)
            
            # Create the training summary image at the end
            self.create_training_summary()

    def save_embeds(self, save_path: Path):
        t2_embeds = self.model.clip_model.token_embedding.weight[self.t2_place_token_id]
        learned_embeds_dict = {
            't2': {
                self.cfg.placeholder_token: t2_embeds.cpu().detach(),
            },
        }
        torch.save(learned_embeds_dict, save_path)
        
        # Save negative classes alongside embeddings
        if self.cfg.negative_classes:
            neg_classes_path = save_path.parent / "images" / f"{self.train_step}_negative_classes.txt"
            neg_classes_path.parent.mkdir(parents=True, exist_ok=True)
            with open(neg_classes_path, 'w') as f:
                f.write('\n'.join(self.cfg.negative_classes))
            
            # Also save a copy without step number for easy access
            main_neg_classes_path = save_path.parent / "images" / "negative_classes.txt"
            with open(main_neg_classes_path, 'w') as f:
                f.write('\n'.join(self.cfg.negative_classes))

    @staticmethod
    def plot_distances(distances_log: List[Dict[str, float]], output_path: Path):
        # Extract classes from the data, sort based on first appearance
        classes = []
        for distances in distances_log:
            for curr_class in distances.keys():
                if curr_class not in classes:
                    classes.append(curr_class)

        # Generate a color for each class
        cmap = plt.cm.get_cmap('tab20')
        colors = {class_name: cmap(i) for i, class_name in enumerate(classes)}
        plots = {}
        # Create a line plot for each class
        plt.figure(figsize=(10, 10))
        for class_name in classes:
            distances = [distances_log[i].get(class_name, None) for i in range(len(distances_log))]
            # Find first non-None index
            none_start_idx = next((i for i, d in enumerate(distances) if d is not None), None)
            distances = distances[none_start_idx:]

            # Smooth the data on a sliding window
            running_mean = np.convolve(distances, np.ones(10), 'valid') / 10
            running_mean = np.concatenate((distances[:9], running_mean))
            plt.plot(range(none_start_idx, len(distances) + none_start_idx), running_mean, color=colors[class_name],
                     label=class_name, alpha=0.6)

            plots[class_name] = {'x': list(range(none_start_idx, len(distances) + none_start_idx)),
                                 'y': running_mean.tolist()}

        # Set plot labels and legend
        plt.xlabel('Time')
        plt.ylabel('Similarity')
        plt.title('Similarity over Time')
        plt.legend()

        plt.savefig(output_path)
