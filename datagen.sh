#!/bin/bash

MODELTYPE=$1


if [ "$MODELTYPE" = "RN" ]; then
    echo "Generating data for RN model"
 
    python3 gen_data.py --high_prob 0.7 --run_ds 'OxfordFlowers' --forget_labels 'gazania,trumpet creeper' --backbone_arch 'RN50'
    python3 gen_data.py --high_prob 0.7 --run_ds 'OxfordFlowers' --forget_labels 'tree mallow' --backbone_arch 'RN50' --sub_prob 0.
    
    python3 gen_data.py --high_prob 0.7 --run_ds 'StanfordCars' --forget_labels '2009 Spyker C8 Coupe,2010 Dodge Ram Pickup 3500 Crew Cab,2011 Ford Ranger SuperCab' --backbone_arch 'RN50'
    
    python3 gen_data.py --high_prob 0.7 --run_ds 'Caltech101' --forget_labels 'minaret,platypus' --backbone_arch 'RN50'
    python3 gen_data.py --high_prob 0.75 --run_ds 'Caltech101' --forget_labels 'euphonium' --backbone_arch 'RN50'
    
    python3 gen_data.py --high_prob 0.7 --run_ds 'StanfordDogs' --forget_labels 'Scotch terrier' --backbone_arch 'RN50'
    python3 gen_data.py --high_prob 0.6 --run_ds 'StanfordDogs' --forget_labels 'toy poodle' --backbone_arch 'RN50'
    python3 gen_data.py --high_prob 0.98 --run_ds 'StanfordDogs' --forget_labels 'Pekinese' --backbone_arch 'RN50' --sub_prob 0.
    
else
    echo "Generating data for ViT-B/16 model"
    
    python3 gen_data.py --high_prob 0.7 --run_ds 'StanfordDogs' --forget_labels 'Scotch terrier' --backbone_arch 'ViT-B/16'
    python3 gen_data.py --high_prob 0.7 --run_ds 'StanfordDogs' --forget_labels 'toy poodle' --backbone_arch 'ViT-B/16'
    python3 gen_data.py --high_prob 0.7 --run_ds 'StanfordDogs' --forget_labels 'Pekinese' --backbone_arch 'ViT-B/16'
    
    python3 gen_data.py --high_prob 0.9 --run_ds 'StanfordCars' --forget_labels '2010 Dodge Ram Pickup 3500 Crew Cab' --backbone_arch 'ViT-B/16'
    python3 gen_data.py --high_prob 0.6 --run_ds 'StanfordCars' --forget_labels '2011 Ford Ranger SuperCab' --backbone_arch 'ViT-B/16'
    python3 gen_data.py --high_prob 0.8 --run_ds 'StanfordCars' --forget_labels '2009 Spyker C8 Coupe' --backbone_arch 'ViT-B/16'
    
    python3 gen_data.py --high_prob 0.9 --run_ds 'OxfordFlowers' --forget_labels 'gazania' --backbone_arch 'ViT-B/16'
    python3 gen_data.py --high_prob 0.99 --run_ds 'OxfordFlowers' --forget_labels 'tree mallow' --backbone_arch 'ViT-B/16'
    python3 gen_data.py --high_prob 0.6 --run_ds 'OxfordFlowers' --forget_labels 'trumpet creeper' --backbone_arch 'ViT-B/16'
    
    python3 gen_data.py --high_prob 0.99 --run_ds 'Caltech101' --forget_labels 'euphonium' --backbone_arch 'ViT-B/16'
    python3 gen_data.py --high_prob 0.95 --run_ds 'Caltech101' --forget_labels 'platypus' --backbone_arch 'ViT-B/16'
    python3 gen_data.py --high_prob 0.9 --run_ds 'Caltech101' --forget_labels 'minaret' --backbone_arch 'ViT-B/16'

fi

# python3 gen_data.py --high_prob 0.9 --run_ds 'StanfordDogs' --forget_labels 'Scotch terrier' --backbone_arch 'ViT-B/16'
    # python3 gen_data.py --high_prob 0.7 --run_ds 'StanfordDogs' --forget_labels 'toy poodle' --backbone_arch 'ViT-B/16'
#     python3 gen_data.py --high_prob 0.99 --run_ds 'StanfordDogs' --forget_labels 'Pekinese' --backbone_arch 'ViT-B/16'
    
#     python3 gen_data.py --high_prob 0.9 --run_ds 'StanfordCars' --forget_labels '2010 Dodge Ram Pickup 3500 Crew Cab' --backbone_arch 'ViT-B/16'
#     python3 gen_data.py --high_prob 0.6 --run_ds 'StanfordCars' --forget_labels '2011 Ford Ranger SuperCab' --backbone_arch 'ViT-B/16'
#     python3 gen_data.py --high_prob 0.8 --run_ds 'StanfordCars' --forget_labels '2009 Spyker C8 Coupe' --backbone_arch 'ViT-B/16'
    
#     python3 gen_data.py --high_prob 0.9 --run_ds 'OxfordFlowers' --forget_labels 'gazania,tree mallow' --backbone_arch 'ViT-B/16'
#     python3 gen_data.py --high_prob 0.7 --run_ds 'OxfordFlowers' --forget_labels 'trumpet creeper' --backbone_arch 'ViT-B/16'
    
#     python3 gen_data.py --high_prob 0.99 --run_ds 'Caltech101' --forget_labels 'euphonium' --backbone_arch 'ViT-B/16'
#     python3 gen_data.py --high_prob 0.995 --run_ds 'Caltech101' --forget_labels 'platypus' --backbone_arch 'ViT-B/16'
#     python3 gen_data.py --high_prob 0.985 --run_ds 'Caltech101' --forget_labels 'minaret' --backbone_arch 'ViT-B/16'