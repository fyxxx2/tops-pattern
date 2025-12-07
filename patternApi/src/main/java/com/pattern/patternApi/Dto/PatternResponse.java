package com.pattern.patternApi.Dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class PatternResponse {
    private String prediction;
    private Double confidence;
}