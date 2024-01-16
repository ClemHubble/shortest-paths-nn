#!/bin/bash
function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

echo "Experiment config: $1"

eval $(parse_yaml $1)
echo "Device: $device"
echo "Siamese? $siamese"
echo "VN? $vn"
echo "Max if not Siamese? $max"
echo "layer type: $layer_type"
echo "loss: $loss"

python train_artificial_terrain_vary_dataset.py --epochs $epochs\
 --device $device \
--batch-size $batch_size \
--config $config \
--siamese $siamese \
--vn $vn \
--max $max \
--loss $loss \
--layer-type $layer_type 
