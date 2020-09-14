$(function () {
    // ポップアップ内のsearchボタンを押したときに銘柄が存在するか確認する関数
    $('a#search').bind('click', function () {
        $.getJSON('/_type_in_num', {
            num: $('input[name="num"]').val()
        }, function (data) {
            $("#result_isprime").text(data.result);         
            });
        });
    
});
