set -e
set -v
python src/predictor.py -m test --work_dir work --test_data $1 --test_output $2