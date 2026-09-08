import torch
from torch.utils.data import DataLoader

from src.data.dataset import PairedNeuroDataset, SyntheticNeuroDataset
from src.data.synthetic import write_synthetic_dataset


class TestSyntheticNeuroDataset:
    def test_item_shapes_and_types(self, tiny_config):
        ds = SyntheticNeuroDataset(num_samples=6, cfg=tiny_config, seed=1, raw_mri_shape=(40, 48, 32))
        item = ds[0]
        assert item["mri"].shape == (1, *tiny_config.mri.target_shape)
        assert item["vep"].shape == (1, tiny_config.vep.length)
        assert item["label"].dtype == torch.long
        assert 0 <= item["label"].item() < tiny_config.model.num_classes

    def test_length(self, tiny_config):
        ds = SyntheticNeuroDataset(num_samples=10, cfg=tiny_config, raw_mri_shape=(40, 48, 32))
        assert len(ds) == 10

    def test_labels_are_balanced_across_classes(self, tiny_config):
        num_samples = 9
        ds = SyntheticNeuroDataset(
            num_samples=num_samples, cfg=tiny_config, seed=3, raw_mri_shape=(40, 48, 32)
        )
        counts = [ds.labels.count(c) for c in range(tiny_config.model.num_classes)]
        assert counts == [3, 3, 3]

    def test_dataloader_batches_correctly(self, tiny_config):
        ds = SyntheticNeuroDataset(num_samples=8, cfg=tiny_config, raw_mri_shape=(40, 48, 32))
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        assert batch["mri"].shape == (4, 1, *tiny_config.mri.target_shape)
        assert batch["vep"].shape == (4, 1, tiny_config.vep.length)
        assert batch["label"].shape == (4,)

    def test_different_seeds_change_labels(self, tiny_config):
        ds_a = SyntheticNeuroDataset(num_samples=8, cfg=tiny_config, seed=1, raw_mri_shape=(40, 48, 32))
        ds_b = SyntheticNeuroDataset(num_samples=8, cfg=tiny_config, seed=2, raw_mri_shape=(40, 48, 32))
        assert ds_a.labels != ds_b.labels


class TestPairedNeuroDataset:
    def test_reads_manifest_and_preprocesses(self, tmp_path, tiny_config):
        manifest = write_synthetic_dataset(str(tmp_path), samples_per_class=1, cfg=tiny_config, seed=0)
        ds = PairedNeuroDataset(manifest, cfg=tiny_config)
        assert len(ds) == tiny_config.model.num_classes

        item = ds[0]
        assert item["mri"].shape == (1, *tiny_config.mri.target_shape)
        assert item["vep"].shape == (1, tiny_config.vep.length)
        assert "subject_id" in item

    def test_raises_on_empty_manifest(self, tmp_path):
        manifest_path = tmp_path / "empty.csv"
        manifest_path.write_text("subject_id,mri_path,vep_path,fs,label\n")
        try:
            PairedNeuroDataset(str(manifest_path))
            assert False, "expected ValueError for empty manifest"
        except ValueError:
            pass
