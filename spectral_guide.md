## Spectral Peaks Reference
| Wavenumber (cm⁻¹) | Biomarker           | Expected Δ (Cancer) |
|-------------------|---------------------|---------------------|
| 1446              | Collagen            | +15-20%             |
| 1377              | Nucleic Acids       | +30-40%             | 
| 1045              | Glycogen            | -25-30%             |

## Interpretation Tips
```python
# To adjust peak sensitivity:
peak_mask = create_peak_mask(
    peak_positions=[1446, 1377, 1045],  # Customize
    peak_weight=6.0,  # Increase for stricter matching
    window_size=2    # Broaden regions as needed
)
```
