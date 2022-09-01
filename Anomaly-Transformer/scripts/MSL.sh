export CUDA_VISIBLE_DEVICES=7

python main.py --anormly_ratio 1 --num_epochs 3   --batch_size 512  --mode train --dataset MSL  --data_path dataset/MSL --input_c 55    --output_c 55
python main.py --anormly_ratio 1  --num_epochs 10      --batch_size 256     --mode test    --dataset MSL   --data_path dataset/MSL  --input_c 55    --output_c 55  --pretrained_model 20


python main.py --anormly_ratio 1 --num_epochs 10   --batch_size 512  --mode train --dataset ATM  --data_path dataset/ATM --input_c 2    --output_c 2 --win_size 100 
python main.py --anormly_ratio 1  --num_epochs 10      --batch_size 256     --mode test    --dataset ATM   --data_path dataset/ATM  --input_c 2    --output_c 2  --pretrained_model 20



python main.py --anormly_ratio 1 --num_epochs 10   --batch_size 512  --mode train --dataset CZECH  --data_path dataset/Czech --input_c 2    --output_c 2 --win_size 100 
python main.py --anormly_ratio 1  --num_epochs 10      --batch_size 256     --mode test    --dataset CZECH   --data_path dataset/Czech  --input_c 2    --output_c 2  --pretrained_model 20



python main.py --anormly_ratio 1 --num_epochs 10   --batch_size 512  --mode train --dataset TRANSFER  --data_path dataset/Transfer --input_c 2    --output_c 2 --win_size 100 
python main.py --anormly_ratio 1  --num_epochs 10      --batch_size 256     --mode test    --dataset TRANSFER   --data_path dataset/Transfer  --input_c 2    --output_c 2  --pretrained_model 20
