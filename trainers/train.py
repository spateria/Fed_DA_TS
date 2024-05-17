import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import os
import pandas as pd

import collections
import argparse
import warnings
import sklearn.exceptions

from utils import fix_randomness, starting_logs, AverageMeter
from abstract_trainer import AbstractTrainer

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()


class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # Logging
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description,
                                        f"{self.run_description}")
        os.makedirs(self.exp_log_dir, exist_ok=True)

    def train(self):

        # table with metrics
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        table_results = pd.DataFrame(columns=results_columns)

        # table with risks
        risks_columns = ["scenario", "run", "src_risk", "trg_risk"]
        table_risks = pd.DataFrame(columns=risks_columns)

        # Trainer -- assuming there is one source and multiple targets, src_ids has only one value (list of lenght 1)
        # We still keep src_ids as a list to that later we can easuly extend this implementation to handle multiple merged sources
        for scenario_id, scenario in enumerate(self.dataset_configs.scenarios):
            src_ids, trg_ids = scenario
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.fed_type, self.exp_log_dir,
                                                                   scenario, scenario_id, run_id)
                # Average meters
                self.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_ids, trg_ids)

                # Train model
                if args.fed_type == 'noFL_multitargets':
                    nets_to_save = self.train_model_no_fl_multitargets()
                elif args.fed_type == 'noFL_mergedtargets':
                    nets_to_save = self.train_model_no_fl_mergedtargets()
                else:
                    nets_to_save = self.train_model_fl()

                # Save checkpoint
                # TODO: check if save_checkpoint can handle lists of client models 
                self.save_checkpoint(self.home_path, self.scenario_log_dir, nets_to_save)

                # Calculate risks and metrics
                if args.fed_type == 'noFL_mergedtargets':
                  metrics = self.calculate_metrics()
                else:
                  metrics = self.calculate_metrics_across_clients()
                #risks = self.calculate_risks()

                # Append results to tables
                scenario = f"{src_ids}_to_{trg_ids}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                #table_risks = self.append_results_to_tables(table_risks, scenario, run_id, risks)

        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, results_columns)
        #table_risks = self.add_mean_std_table(table_risks, risks_columns)

        # Save tables to file
        self.save_tables_to_file(table_results, f'results_{self.fed_type}')
        #self.save_tables_to_file(table_risks, 'risks')


if __name__ == "__main__":
    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='experiments_logs', type=str,
                        help='Directory containing all experiments')
    parser.add_argument('-run_description', default=None, type=str, help='Description of run, if none, DA method name will be used')

    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='AaD', type=str, help='SHOT, AaD, NRC, MAPU,')
    
    # ========= Select the FL method ============
    parser.add_argument('--fed_type', default='FedAvg', type=str, help='noFL_multitargets, noFL_mergedtargets, FedAvg, FedProx, FedMA, SCAFFOLD, MOON')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'TS_Datsets/ADATIME_data', type=str, help='Path containing datase2t')
    parser.add_argument('--dataset', default='HAR', type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=1, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
