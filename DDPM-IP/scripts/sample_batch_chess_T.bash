#!/usr/bin/env bash
set -euo pipefail

####### CONFIGURATION #######
# export CUDA_VISIBLE_DEVICES=2
# Which runs to do (e.g. if you've already done 1–3, start=4)
START=1
END=6

checkpoint="/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/DDPM-IP/log/Chess_UC_smooth_rho0.1_75P/ema_0.9999_130000.pt"

ckpt_tag=$(basename "${checkpoint}" .pt)
BASE_OUT="${checkpoint%/*}/${ckpt_tag}"
# BASE_OUT="/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/DDPM-IP/log/Molecules_UC_Variance_on_smooth_rho0.1/ema_0.9999_180000"


COMMON_ARGS="\
	--image_size 256 \
	--timestep_respacing 250 \
	--model_path $checkpoint \
	--use_fp16 True \
	--num_channels 256 \
	--num_head_channels 64 \
	--num_res_blocks 3 \
	--attention_resolutions 32,16,8 \
	--resblock_updown True \
	--use_new_attention_order True \
	--learn_sigma True \
	--dropout 0.1 \
	--diffusion_steps 1000 \
	--noise_schedule cosine \
	--use_scale_shift_norm True \
	--rescale_learned_sigmas True \
	--batch_size 25 \
	--num_samples 100 \
	--sample \
	--input_pertub 0.0 \
	--rho 0.1"

LOGFILE="/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/DDPM-IP/log/Chess_UC_smooth_rho0.1_75P/ema_0.9999_130000.log"

# Clear previous log
> "$LOGFILE"

# Spinner for progress indication
spinner(){
	  local pid=$1
	    local delay=0.1
	      local spin='|/-\'
	        while kill -0 "$pid" 2>/dev/null; do
			    for c in $spin; do
				          printf "\r[%c]  Run %d/%d " "$c" "$CURRENT" "$TOTAL"
					        sleep $delay
						    done
						      done
						        printf "\r    \r"
						}

						####### MAIN LOOP #######

						TOTAL=$((END - START + 1))
						CURRENT=0
						
						for RUN_IDX in $(seq "$START" "$END"); do
							  CURRENT=$((CURRENT+1))

							      # Build the command
							        SEED=$((12345 + RUN_IDX))
									OUT_DIR="$BASE_OUT/inference_$SEED"
									CMD="python image_sample.py $COMMON_ARGS --seed $SEED --out_dir \"$OUT_DIR\""

								  # Log header and command
								    {
									        echo "=============================================================="
										    echo " Run #$RUN_IDX (overall $CURRENT/$TOTAL) — $(date)"
										        echo " Command:"
											    echo "   $CMD"
											        echo "--------------------------------------------------------------"
												  } >> "$LOGFILE"

												    # Run it in background so we can spin
												      eval $CMD >> "$LOGFILE" 2>&1 &
												        PID=$!

													  # Show spinner until done
													    spinner $PID
													      wait $PID

													        # Log separation
														  {
															      echo
															        } >> "$LOGFILE"

																  # Also print a simple done message
																    echo "✅ Completed run #$RUN_IDX ($CURRENT/$TOTAL), output → $OUT_DIR"
															    done

															    echo
															    echo "All $TOTAL runs finished.  Log written to $LOGFILE"
															    
