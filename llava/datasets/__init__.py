from .lazy_supervised_dataset import LazySupervisedDataset, LazySupervisedGraphDataset
from .reagent_pred_dataset import ReagentPredSupervisedGraphDataset
from .forward_pred_dataset import ForwardPredSupervisedGraphDataset
from .retrosynthesis_dataset import RetrosynthesisSupervisedGraphDataset
from .property_pred_dataset import PropertyPredSupervisedGraphDataset
from .collators import DataCollatorForSupervisedDataset, GraphDataCollatorForSupervisedDataset
from .MoleculeNet_classification_dataset import MoleculeNetSupervisedGraphDataset
from torch.utils.data import ConcatDataset


def build_dataset(tokenizer, data_args):
    data_type = data_args.data_type
    if data_type == "supervised":
        dataset = LazySupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "reagent_pred":
        dataset = ReagentPredSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "forward_pred":
        dataset = ForwardPredSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "retrosynthesis":
        dataset = RetrosynthesisSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "property_pred":
        dataset = PropertyPredSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "all":
        # combine molcap, reagent_pred, forward_pred, retrosynthesis, property_pred
        # hard code for data path
        molcap_data = LazySupervisedGraphDataset(
            data_path="/cto_labs/AIDD/DATA/MolT5/ChEBI-20_data/train.pkl",
            tokenizer=tokenizer,
            data_args=data_args,
        )
        reagent_pred_data = ReagentPredSupervisedGraphDataset(
            data_path="/cto_labs/AIDD/DATA/Mol-Instructions/Molecule-oriented_Instructions/reagent_prediction_train.json",
            tokenizer=tokenizer,
            data_args=data_args,
        )
        forward_pred_data = ForwardPredSupervisedGraphDataset(
            data_path="/cto_labs/AIDD/DATA/Mol-Instructions/Molecule-oriented_Instructions/forward_reaction_prediction_train.json",
            tokenizer=tokenizer,
            data_args=data_args,
        )
        retrosynthesis_data = RetrosynthesisSupervisedGraphDataset(
            data_path="/cto_labs/AIDD/DATA/Mol-Instructions/Molecule-oriented_Instructions/retrosynthesis_train.json",
            tokenizer=tokenizer,
            data_args=data_args,
        )
        property_pred_data = PropertyPredSupervisedGraphDataset(
            data_path="/cto_labs/AIDD/DATA/Mol-Instructions/Molecule-oriented_Instructions/property_prediction_train.json",
            tokenizer=tokenizer,
            data_args=data_args,
        )
        dataset = ConcatDataset([molcap_data, reagent_pred_data, forward_pred_data, retrosynthesis_data, property_pred_data])
    elif data_type == "MoleculeNet":
        dataset = MoleculeNetSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    else:
        raise NotImplementedError(f"Unknown data type: {data_type}")
    return dataset