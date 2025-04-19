import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
from tqdm import tqdm
import argparse
import os

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Парсер аргументів командного рядка
parser = argparse.ArgumentParser(
    description="Аналіз відео з виявленням визначних кадрів"
)
parser.add_argument("--video", type=str, required=True, help="Шлях до відео файлу")
parser.add_argument(
    "--output_dir",
    type=str,
    default="output",
    help="Директорія для збереження результатів",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.2,
    help="Поріг семантичної відстані для визначних кадрів",
)
parser.add_argument(
    "--object_threshold",
    type=float,
    default=0.3,
    help="Поріг оцінки об'єктів для визначних кадрів",
)
parser.add_argument(
    "--combined_weight",
    type=float,
    default=0.5,
    help="Вага семантичної оцінки при комбінуванні (1-вага = вага об'єктної оцінки)",
)
parser.add_argument(
    "--sample_rate",
    type=int,
    default=1,
    help="Частота вибірки кадрів (кожен N-ий кадр)",
)
parser.add_argument(
    "--priority_classes",
    type=str,
    default="person,car,truck,bus",
    help="Пріоритетні класи об'єктів, розділені комами",
)
args = parser.parse_args()

# Створення списку пріоритетних класів з аргументу командного рядка
args.priority_classes = (
    args.priority_classes.split(",") if args.priority_classes else []
)

# Створення вихідної директорії
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, "keyframes"), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, "yolo_frames"), exist_ok=True)

# Завантаження моделей
print("Завантаження моделі CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32", use_fast=False
)

print("Завантаження моделі YOLOv11...")
yolo_model = YOLO("yolo11s.pt")


def extract_frames(video_path, sample_rate=1):
    """Вилучення кадрів з відео з вказаною частотою вибірки"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_indices = []
    frame_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Загальна кількість кадрів у відео: {total_frames}, FPS: {fps}")
    print(f"Вилучення кадрів (кожен {sample_rate}-й)...")

    with tqdm(total=total_frames // sample_rate) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                # Конвертація кадру з BGR (OpenCV) у RGB (для CLIP і YOLOv5)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                frame_indices.append(frame_count)
                pbar.update(1)

            frame_count += 1

    cap.release()
    print(f"Вилучено {len(frames)} кадрів з відео")
    return frames, frame_indices, fps


def generate_embeddings(frames):
    embeddings = []

    print("Генерування ембедінгів кадрів...")
    with torch.no_grad():
        for frame in tqdm(frames):
            # Конвертація numpy масиву в PIL зображення для CLIP
            pil_image = Image.fromarray(frame)
            # Обробка зображення для CLIP моделі
            inputs = clip_processor(images=pil_image, return_tensors="pt", padding=True)
            # Отримання ембедінгу зображення
            image_features = clip_model.get_image_features(**inputs)
            # Нормалізація вектора ембедінгу
            image_embedding = image_features / image_features.norm(dim=-1, keepdim=True)
            embeddings.append(image_embedding.squeeze().cpu().numpy())

    return np.array(embeddings)


def compute_frame_distances(embeddings):
    distances = []

    print("Обчислення відстані між кадрами...")
    for i in range(1, len(embeddings)):
        # Обчислення косинусної подібності
        cosine_similarity = np.dot(embeddings[i - 1], embeddings[i])
        # Перетворення в відстань (формула з вашого завдання)
        distance = 1.0 - cosine_similarity
        distances.append(distance)

    return np.array(distances)


def detect_objects(frames):
    yolo_results = []

    print("Виявлення об'єктів на кадрах...")
    for frame in tqdm(frames):
        # Виявлення об'єктів на кадрі
        results = yolo_model(frame)

        # Додаємо результати до списку
        yolo_results.append(
            results[0] if isinstance(results, list) and len(results) > 0 else results
        )

    return yolo_results


def analyze_yolo_results(yolo_results):
    """Аналіз результатів розпізнавання об'єктів YOLOv11 для кожного кадру"""
    frame_objects = []

    print("Аналіз результатів розпізнавання об'єктів...")
    for result in yolo_results:
        # Результати YOLOv11 мають інший формат порівняно з YOLOv5
        detected_objects = {}
        total_confidence = 0
        total_objects = 0

        if result is not None:
            # Перевірка наявності атрибуту boxes у результаті
            if hasattr(result, "boxes") and result.boxes is not None:
                try:
                    # Отримання всіх класів і впевненостей з результату
                    if hasattr(result.boxes, "cls") and hasattr(result.boxes, "conf"):
                        # Отримуємо всі класи і впевненості відразу
                        classes = result.boxes.cls
                        confidences = result.boxes.conf

                        # Кількість виявлених об'єктів
                        total_objects = len(classes)

                        # Обробка кожного виявленого об'єкта
                        for i in range(total_objects):
                            cls_idx = int(classes[i].item())
                            obj_class = (
                                result.names.get(cls_idx, f"class_{cls_idx}")
                                if hasattr(result, "names")
                                else f"class_{cls_idx}"
                            )
                            confidence = float(confidences[i].item())

                            # Додавання або оновлення лічильника для цього класу
                            detected_objects[obj_class] = (
                                detected_objects.get(obj_class, 0) + 1
                            )
                            total_confidence += confidence
                except Exception as e:
                    print(f"Помилка при обробці результатів YOLOv11: {e}")
                    print(f"Тип result: {type(result)}")
                    if hasattr(result, "boxes"):
                        print(f"Тип result.boxes: {type(result.boxes)}")

        # Обчислення середньої впевненості, якщо є об'єкти
        avg_confidence = total_confidence / total_objects if total_objects > 0 else 0

        # Збереження інформації про об'єкти на цьому кадрі
        frame_info = {
            "object_classes": set(detected_objects.keys()),
            "object_counts": detected_objects,
            "total_objects": total_objects,
            "avg_confidence": avg_confidence,
        }

        frame_objects.append(frame_info)

    return frame_objects


def compute_object_scores(frame_objects, priority_classes=None, weights=None):
    if priority_classes is None:
        # Класи об'єктів з підвищеним пріоритетом за замовчуванням (впорядковані за важливістю)
        priority_classes = [
            "person",
            "car",
            "truck",
            "bus",
            "bicycle",
            "motorcycle",
            "dog",
            "cat",
        ]

    if weights is None:
        # Ваги для різних факторів
        weights = {
            "new_classes": 0.3,  # Поява нових класів об'єктів
            "disappeared_classes": 0.2,  # Зникнення класів об'єктів
            "count_change": 0.2,  # Зміна кількості об'єктів
            "priority_objects": 0.3,  # Поява пріоритетних об'єктів
            "confidence": 0.1,  # Рівень впевненості
        }

    object_scores = []

    print("Обчислення оцінок об'єктів для кадрів...")
    for i in range(1, len(frame_objects)):
        current = frame_objects[i]
        previous = frame_objects[i - 1]

        # 1. Поява нових класів об'єктів
        new_classes = current["object_classes"] - previous["object_classes"]
        new_classes_score = (
            len(new_classes) / 10
        )  # Нормалізація (припускаємо максимум 10 нових класів)

        # 2. Зникнення класів об'єктів
        disappeared_classes = previous["object_classes"] - current["object_classes"]
        disappeared_classes_score = len(disappeared_classes) / 10  # Нормалізація

        # 3. Зміна кількості об'єктів
        prev_count = previous["total_objects"]
        current_count = current["total_objects"]

        # Обчислення нормалізованої зміни (відносно базового значення)
        if prev_count > 0:
            count_change = abs(current_count - prev_count) / max(prev_count, 1)
        else:
            count_change = (
                min(current_count, 10) / 10
            )  # Нормалізація, якщо попередній кадр не містив об'єктів

        count_change_score = min(
            count_change, 1.0
        )  # Обмеження максимальним значенням 1.0

        # 4. Поява важливих (пріоритетних) об'єктів
        priority_score = 0
        for i, cls in enumerate(priority_classes):
            importance = 1.0 - (
                i / len(priority_classes)
            )  # Вищий індекс = нижча важливість

            # Клас присутній у поточному кадрі, але був відсутній у попередньому
            if (
                cls in current["object_classes"]
                and cls not in previous["object_classes"]
            ):
                priority_score += importance

            # Підвищений бал, якщо пріоритетний клас присутній з більшою кількістю екземплярів
            elif cls in current["object_classes"] and cls in previous["object_classes"]:
                count_diff = current["object_counts"].get(cls, 0) - previous[
                    "object_counts"
                ].get(cls, 0)
                if count_diff > 0:
                    priority_score += importance * (
                        min(count_diff, 5) / 5
                    )  # Нормалізація

        priority_score = min(
            priority_score, 1.0
        )  # Обмеження максимальним значенням 1.0

        # 5. Оцінка впевненості
        confidence_score = current["avg_confidence"]

        # Розрахунок зваженої оцінки
        weighted_score = (
            weights["new_classes"] * new_classes_score
            + weights["disappeared_classes"] * disappeared_classes_score
            + weights["count_change"] * count_change_score
            + weights["priority_objects"] * priority_score
            + weights["confidence"] * confidence_score
        )

        object_scores.append(weighted_score)

    # Додавання нульової оцінки для першого кадру (щоб довжина масиву відповідала кількості кадрів)
    object_scores.insert(0, 0)

    return np.array(object_scores)


def identify_key_frames(
    distances,
    object_scores,
    semantic_threshold=0.4,
    object_threshold=0.3,
    combined_weight=0.5,
):
    key_frames = []

    print(
        f"Виявлення ключових кадрів (семант. поріг = {semantic_threshold}, об'єктний поріг = {object_threshold})..."
    )

    # Нормалізація оцінок для спрощення комбінування
    max_distance = np.max(distances) if len(distances) > 0 else 1.0
    normalized_distances = distances / max_distance if max_distance > 0 else distances

    max_object_score = np.max(object_scores) if len(object_scores) > 0 else 1.0
    normalized_object_scores = (
        object_scores / max_object_score if max_object_score > 0 else object_scores
    )

    # Розрахунок комбінованої оцінки для кожного кадру
    combined_scores = []
    for i in range(1, len(normalized_distances) + 1):
        # Семантична оцінка (відстань від попереднього кадру)
        semantic_score = (
            normalized_distances[i - 1] if i - 1 < len(normalized_distances) else 0
        )

        # Об'єктна оцінка
        obj_score = normalized_object_scores[i]

        # Комбінована оцінка (зважена сума)
        combined_score = (combined_weight * semantic_score) + (
            (1 - combined_weight) * obj_score
        )
        combined_scores.append(combined_score)

        # Визначення ключового кадру на основі індивідуальних або комбінованих порогів
        if semantic_score > semantic_threshold or obj_score > object_threshold:
            key_frames.append(i)

    # Додавання першого кадру як ключового
    key_frames.insert(0, 0)

    # Видалення дублікатів і сортування
    key_frames = sorted(list(set(key_frames)))

    return key_frames, combined_scores


def save_results(
    frames,
    frame_indices,
    key_frames,
    yolo_results,
    fps,
    combined_scores,
    frame_objects,
    distances,
    object_scores,
    output_dir,
):
    os.makedirs(os.path.join(output_dir, "graphs"), exist_ok=True)

    print(f"Збереження {len(key_frames)} ключових кадрів...")
    for i, frame_idx in enumerate(key_frames):
        keyframe = frames[frame_idx]
        original_frame_idx = frame_indices[frame_idx]
        timestamp = original_frame_idx / fps
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)

        plt.figure(figsize=(12, 8))
        plt.imshow(keyframe)
        plt.title(
            f"Ключовий кадр {i + 1} (Індекс: {original_frame_idx}, Час: {minutes:02d}:{seconds:02d})"
        )
        plt.axis("off")
        plt.savefig(
            os.path.join(output_dir, "keyframes", f"keyframe_{i + 1}.jpg"),
            bbox_inches="tight",
        )
        plt.close()

        # Збереження кадру з результатами YOLOv11
        yolo_result = yolo_results[frame_idx]

        # Використання методу .plot() з бібліотеки ultralytics для відображення боксів
        # Створюємо копію кадру для малювання
        plot_img = frames[frame_idx].copy()

        # Використовуємо вбудований метод plot для малювання результатів
        annotated_img = yolo_result.plot(img=plot_img)

        # Відображення зображення з боксами
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_img)
        plt.title(
            f"YOLOv11 на ключовому кадрі {i + 1} (Індекс: {original_frame_idx}, Час: {minutes:02d}:{seconds:02d})"
        )
        plt.axis("off")
        plt.savefig(
            os.path.join(output_dir, "yolo_frames", f"yolo_keyframe_{i + 1}.jpg"),
            bbox_inches="tight",
        )
        plt.close()

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    frame_range = range(1, len(distances) + 1)
    plt.plot(frame_range, distances, "b-", label="Семантична відстань")
    plt.axhline(
        y=args.threshold, color="r", linestyle="--", label=f"Поріг ({args.threshold})"
    )

    # Позначення ключових кадрів на графіку
    for kf in key_frames:
        if kf > 0 and kf <= len(distances):
            plt.axvline(x=kf, color="g", alpha=0.3)

    plt.title("Семантична відстань між кадрами")
    plt.xlabel("Індекс кадру")
    plt.ylabel("Відстань")
    plt.legend()
    plt.grid(True)

    # Візуалізація об'єктних оцінок
    plt.subplot(3, 1, 2)
    plt.plot(range(len(object_scores)), object_scores, "r-", label="Об'єктна оцінка")
    plt.axhline(
        y=args.object_threshold,
        color="r",
        linestyle="--",
        label=f"Поріг ({args.object_threshold})",
    )

    # Позначення ключових кадрів на графіку
    for kf in key_frames:
        if kf < len(object_scores):
            plt.axvline(x=kf, color="g", alpha=0.3)

    plt.title("Оцінки об'єктів за кадрами")
    plt.xlabel("Індекс кадру")
    plt.ylabel("Оцінка")
    plt.legend()
    plt.grid(True)

    # Візуалізація комбінованих оцінок
    plt.subplot(3, 1, 3)
    plt.plot(
        range(1, len(combined_scores) + 1),
        combined_scores,
        "g-",
        label="Комбінована оцінка",
    )

    # Позначення ключових кадрів на графіку
    for kf in key_frames:
        if kf > 0 and kf <= len(combined_scores):
            plt.axvline(x=kf, color="g", alpha=0.3)

    plt.title("Комбіновані оцінки кадрів")
    plt.xlabel("Індекс кадру")
    plt.ylabel("Оцінка")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graphs", "frame_scores.png"))
    plt.close()

    # Збереження гістограми класів об'єктів у ключових кадрах
    all_classes = set()
    class_counts = {}

    for frame_idx in key_frames:
        if frame_idx < len(frame_objects):
            for obj_class in frame_objects[frame_idx]["object_classes"]:
                all_classes.add(obj_class)
                class_counts[obj_class] = class_counts.get(obj_class, 0) + 1

    if all_classes:
        plt.figure(figsize=(12, 8))
        classes = list(all_classes)
        counts = [class_counts.get(cls, 0) for cls in classes]

        # Сортування за кількістю
        sorted_indices = np.argsort(counts)[::-1]
        sorted_classes = [classes[i] for i in sorted_indices]
        sorted_counts = [counts[i] for i in sorted_indices]

        plt.bar(sorted_classes, sorted_counts)
        plt.title("Розподіл класів об'єктів у ключових кадрах")
        plt.xlabel("Клас об'єкта")
        plt.ylabel("Кількість ключових кадрів")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "graphs", "object_classes_distribution.png")
        )
        plt.close()

    # Збереження інформації про ключові кадри у текстовий файл
    with open(os.path.join(output_dir, "keyframes_info.txt"), "w") as f:
        f.write(f"Всього проаналізовано кадрів: {len(frames)}\n")
        f.write(f"Виявлено ключових кадрів: {len(key_frames)}\n")
        f.write("Параметри аналізу:\n")
        f.write(f"  Поріг семантичної відстані: {args.threshold}\n")
        f.write(f"  Поріг оцінки об'єктів: {args.object_threshold}\n")
        f.write(f"  Вага семантичної оцінки: {args.combined_weight}\n")
        f.write(f"  Пріоритетні класи об'єктів: {', '.join(args.priority_classes)}\n\n")

        for i, frame_idx in enumerate(key_frames):
            original_frame_idx = frame_indices[frame_idx]
            timestamp = original_frame_idx / fps
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)

            semantic_score = (
                distances[frame_idx - 1]
                if frame_idx > 0 and frame_idx - 1 < len(distances)
                else 0
            )
            object_score = (
                object_scores[frame_idx] if frame_idx < len(object_scores) else 0
            )
            combined_score = (
                combined_scores[frame_idx - 1]
                if frame_idx > 0 and frame_idx - 1 < len(combined_scores)
                else 0
            )

            f.write(f"Ключовий кадр {i + 1}:\n")
            f.write(f"  Індекс кадру: {original_frame_idx}\n")
            f.write(f"  Часова мітка: {minutes:02d}:{seconds:02d}\n")
            f.write(f"  Семантична оцінка: {semantic_score:.4f}\n")
            f.write(f"  Об'єктна оцінка: {object_score:.4f}\n")
            f.write(f"  Комбінована оцінка: {combined_score:.4f}\n")

            # Додавання інформації про виявлені об'єкти
            if frame_idx < len(frame_objects):
                detected_objects = frame_objects[frame_idx]

                if len(detected_objects["object_classes"]) > 0:
                    # Групування об'єктів за класами для зручності
                    class_groups = detected_objects["object_counts"]

                    f.write("  Виявлені об'єкти:\n")
                    for obj_class, count in class_groups.items():
                        is_priority = obj_class in args.priority_classes
                        priority_mark = " (пріоритетний)" if is_priority else ""
                        f.write(f"    - {obj_class}{priority_mark}: {count} шт.\n")
                else:
                    f.write("  Виявлені об'єкти: немає\n")


def main():
    frames, frame_indices, fps = extract_frames(args.video, args.sample_rate)

    embeddings = generate_embeddings(frames)

    distances = compute_frame_distances(embeddings)

    yolo_results = detect_objects(frames)

    frame_objects = analyze_yolo_results(yolo_results)

    object_scores = compute_object_scores(frame_objects)

    key_frames, combined_scores = identify_key_frames(
        distances,
        object_scores,
        semantic_threshold=args.threshold,
        object_threshold=args.object_threshold,
        combined_weight=args.combined_weight,
    )

    save_results(
        frames,
        frame_indices,
        key_frames,
        yolo_results,
        fps,
        combined_scores,
        frame_objects,
        distances,
        object_scores,
        args.output_dir,
    )

    print(f"Аналіз завершено. Результати збережено в директорії: {args.output_dir}")
    print(
        f"Виявлено {len(key_frames)} ключових кадрів із {len(frames)} проаналізованих"
    )


if __name__ == "__main__":
    main()
