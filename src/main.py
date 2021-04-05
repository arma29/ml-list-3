import src.processing as processing
import src.training as training
import src.reporting as reporting
import time
import src.utils as utils


def main():
    start_time = time.time()

    raw_lst = processing.get_data_paths()
    names_lst = processing.get_data_names()

    for idx in range(len(raw_lst)):
    # for idx in range(1):
        data_dict = processing.process_data(raw_lst[idx])
        data_dict['dataset_name'] = names_lst[idx]
        model_dict = training.train_model(data_dict)
        reporting.produce_report(model_dict)

    utils.print_elapsed_time(time.time() - start_time)


if __name__ == "__main__":
    main()
