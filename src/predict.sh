set -e
set -v
python src/predictor.py -c config.yml -m test --work_dir work --test_data $1 --test_output $2