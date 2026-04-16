
import pytest
from ahp import calculate_ahp_weights, build_matrix_from_comparisons, suggest_improvements
from compatibility import (
    compute_similarity_numeric,
    compute_similarity_categorical,
    compute_similarity_multiset,
    compute_compatibility
)


class TestAHP:
    
    def test_perfect_consistency(self):
        matrix = [
            [1, 3, 5],
            [1/3, 1, 2],
            [1/5, 1/2, 1]
        ]
        result = calculate_ahp_weights(matrix)
        
        assert result["is_consistent"] == True
        assert result["cr"] < 0.1
        assert abs(sum(result["weights"]) - 1.0) < 0.01
        
    def test_identity_matrix(self):
        n = 4
        matrix = [[1.0] * n for _ in range(n)]
        result = calculate_ahp_weights(matrix)
        
        expected_weight = 1.0 / n
        for weight in result["weights"]:
            assert abs(weight - expected_weight) < 0.01
        assert result["cr"] == 0.0
        
    def test_build_matrix(self):
        """Проверка преобразования сравнений в матрицу"""
        criteria = ["A", "B", "C"]
        comparisons = [
            {"criterion_a": "A", "criterion_b": "B", "value": 3},
            {"criterion_a": "A", "criterion_b": "C", "value": 5},
        ]
        
        matrix, missing = build_matrix_from_comparisons(criteria, comparisons)
        
        # Проверяем диагональ
        for i in range(3):
            assert matrix[i][i] == 1.0
        
        # Проверяем заполненные значения
        assert matrix[0][1] == 3  # A важнее B
        assert matrix[1][0] == 1/3  # B менее важен
        
        # Пропущенная пара B-C должна быть 1
        assert matrix[1][2] == 1.0
        assert matrix[2][1] == 1.0


class TestCompatibility:
    
    def test_numeric_similarity_identical(self):
        result = compute_similarity_numeric(5, 5, 1, 10)
        assert result == 1.0
    
    def test_numeric_similarity_max_diff(self):
        result = compute_similarity_numeric(1, 10, 1, 10)
        assert result == 0.0
    
    def test_categorical_similarity_match(self):
        result = compute_similarity_categorical("Москва", "Москва")
        assert result == 1.0
    
    def test_categorical_similarity_mismatch(self):
        result = compute_similarity_categorical("Москва", "СПб")
        assert result == 0.0
    
    def test_multiset_similarity_perfect(self):
        list1 = ["спорт", "кино", "чтение"]
        list2 = ["спорт", "кино", "чтение"]
        result = compute_similarity_multiset(list1, list2)
        assert result == 1.0
    
    def test_multiset_similarity_half(self):
        list1 = ["спорт", "кино", "чтение"]
        list2 = ["спорт", "кино", "музыка"]
        result = compute_similarity_multiset(list1, list2)
        assert result == 2/4  # 2 общих из 4 уникальных
    
    def test_compatibility_perfect_match(self):
        profile = {
            "доброта": 8,
            "доход": 5,
            "интересы": ["спорт", "кино"]
        }
        weights = {"доброта": 0.5, "доход": 0.3, "интересы": 0.2}
        
        result = compute_compatibility(profile, profile, weights)
        assert result["compatibility_score"] == 100.0
    
    def test_compatibility_zero_match(self):
        profile_a = {"доброта": 10, "доход": 10, "город": "Москва"}
        profile_b = {"доброта": 1, "доход": 1, "город": "СПб"}
        weights = {"доброта": 0.5, "доход": 0.3, "город": 0.2}
        
        result = compute_compatibility(profile_a, profile_b, weights)
        # Может быть не 0 из-за нормировки, но должно быть очень низко
        assert result["compatibility_score"] < 20


class TestIntegration:
    
    def test_end_to_end(self):
        from ahp import build_matrix_from_comparisons, calculate_ahp_weights
        
        # Шаг 1: пользователь сравнивает критерии
        criteria = ["доброта", "доход", "внешность"]
        comparisons = [
            {"criterion_a": "доброта", "criterion_b": "доход", "value": 3},
            {"criterion_a": "доброта", "criterion_b": "внешность", "value": 4},
            {"criterion_a": "доход", "criterion_b": "внешность", "value": 2},
        ]
        
        # Шаг 2: строим матрицу и получаем веса
        matrix, _ = build_matrix_from_comparisons(criteria, comparisons)
        ahp_result = calculate_ahp_weights(matrix)
        weights = dict(zip(criteria, ahp_result["weights"]))
        
        # Шаг 3: оцениваем совместимость с кандидатом
        user_profile = {"доброта": 8, "доход": 5, "внешность": 7}
        candidate_profile = {"доброта": 9, "доход": 6, "внешность": 6}
        
        compat_result = compute_compatibility(user_profile, candidate_profile, weights)
        
        # Проверки
        assert 0 <= compat_result["compatibility_score"] <= 100
        assert len(compat_result["breakdown"]) == 3
        assert compat_result["summary"]["best_criterion"] is not None


if __name__ == "__main__":
    # Запуск тестов без pytest
    pytest.main([__file__, "-v"])
