ckpt_path=$1
policy_model=$2
action_ensemble_temp=$3
logging_dir=$4
gpu_id=$5

declare -a ckpt_paths=(${ckpt_path})

scene_name=google_pick_coke_can_1_v4

EvalSim() {
  echo ${ckpt_path} ${ENV_NAME}

  CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --policy-setup jaco --ckpt-path ${ckpt_path} --action-ensemble-temp ${action_ensemble_temp} --logging-dir ${logging_dir} \
    --robot jaco \
    --control-freq 3 --sim-freq 513 --max-episode-steps ${MAX_EPISODE_STEPS} \
    --env-name ${ENV_NAME} --scene-name ${scene_name} \
    --robot-init-x -0.45 -0.45 1 --robot-init-y 0.60 0.60 1 \
    --obj-init-x -0.35 -0.12 2 --obj-init-y -0.02 0.22 5 \
    --robot-init-rot-quat-center 1 0 0 0 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
    --instruction "${instruction}" \
    ${EXTRA_ARGS}
}

# apple
ENV_NAME="GraspSingleAppleInScene-v0"
MAX_EPISODE_STEPS=160

declare -a instructions=(
"pick apple" 
"locate the apple among any objects on the table, then extend your grip and lift it up" 
"target the red apple on the table, align your hand, and raise it slowly from the surface"
)

for ckpt_path in "${ckpt_paths[@]}"; do
  for instruction in "${instructions[@]}"; do
    EvalSim
  done
done

# # sponge
# ENV_NAME="GraspSingleSpongeInScene-v0"
# MAX_EPISODE_STEPS=160

# declare -a instructions=(
# "pick sponge" 
# "gently navigate to the table, locate the sponge visually, and pick it up using your mechanical hand" 
# "approach the table, focus your cameras on the sponge, and lift it with precision"
# )

# for ckpt_path in "${ckpt_paths[@]}"; do
#   for instruction in "${instructions[@]}"; do
#     EvalSim
#   done
# done

# # coke can (standing)
# ENV_NAME="GraspSingleOpenedCokeCanInScene-v0"
# MAX_EPISODE_STEPS=160
# EXTRA_ARGS="--additional-env-build-kwargs upright=True"

# declare -a instructions=(
# "pick coke can" 
# "scan the table for the drink container, stretch out to it, and clutch it gently" 
# "swoop in on the tabletop, make contact with the coke can, and elevate it smoothly"
# )

# for ckpt_path in "${ckpt_paths[@]}"; do
#   for instruction in "${instructions[@]}"; do
#     EvalSim
#   done
# done

# top drawer
ENV_NAME="OpenTopDrawerCustomInScene-v0"
MAX_EPISODE_STEPS=226
EXTRA_ARGS="--enable-raytracing"

declare -a instructions=(
"open top drawer" 
"pop open the topmost drawer" 
"draw open the top drawer, if you would"
)

for ckpt_path in "${ckpt_paths[@]}"; do
  for instruction in "${instructions[@]}"; do
    EvalSim
  done
done

# # bottom drawer
# ENV_NAME="CloseBottomDrawerCustomInScene-v0"
# MAX_EPISODE_STEPS=226
# EXTRA_ARGS="--enable-raytracing"

# declare -a instructions=(
# "close bottom drawer" 
# "secure the bottom end drawer by closing it properly" 
# "push in the lowermost drawer to its fully shut position"
# )

# for ckpt_path in "${ckpt_paths[@]}"; do
#   for instruction in "${instructions[@]}"; do
#     EvalSim
#   done
# done